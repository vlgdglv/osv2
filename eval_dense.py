import os
import sys
import lmdb
import json
import torch
import torch.nn as nn
import logging
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    AutoTokenizer, PreTrainedTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding
)

from model_zoo import BiEncoder
from tqdm import tqdm
from inverted_index import InvertedIndex
from auxiliary import (
    to_device, load_gt, compute_metrics,
    DataConfig, 
    ModelConfig, 
    EvaluationConfig, 
    write_fbin, write_ibin, 
    read_fbin, read_ibin
)
import faiss

logger = logging.getLogger(__name__)

class MARCOWSTestIdsDataset(Dataset):
    def __init__(self, passage_lmdb_env, tokenizer, 
                 start_idx=0, end_idx=-1,
                 idmapping=None, max_length=128):
        self.passage_lmdb_env = passage_lmdb_env 
        self.start = start_idx
        self.end = end_idx
        self.length = self.end - self.start
        if idmapping:
            self.idmapping = idmapping
            self.mapper = lambda x: self.idmapping[str(x)]
        else:
            self.mapper = lambda x: x+1 # For train
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        real_index = str(self.mapper(self.start + index))
        with self.passage_lmdb_env.begin(write=False) as doc_pool_txn:
            text_ids = pickle.loads(doc_pool_txn.get(real_index.encode()))
        encoded_text = self.tokenizer.prepare_for_model(
            text_ids,
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            add_special_tokens=True
        )
        return real_index, encoded_text

@dataclass
class PredictionCollator(DataCollatorWithPadding):
    is_query: bool = False
    q_max_len: int = 32
    k_max_len: int = 128
    def __call__(self, features):
        text_ids, encode_texts = [f[0] for f in features], [f[1] for f in features]
        collated_texts = super().__call__(encode_texts)
        return text_ids, collated_texts

class TextDatasetLMDBMeta(torch.utils.data.Dataset):
    def __init__(self, start_idx, end_idx, doc_pool_txn, tokenizer, id2id=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.doc_pool_txn = doc_pool_txn
        self.length = self.end_idx - self.start_idx
        self.tokenizer = tokenizer
        self.max_seq_length = 128
        self.id2id = id2id

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.id2id!=None:
            real_index = str(self.id2id[str(index + self.start_idx)])
        else:
            real_index = str(index + 1 + self.start_idx)
        url, title, body = pickle.loads(self.doc_pool_txn.get(real_index.encode()))

        prev_tokens = ['[CLS]']  + self.tokenizer.tokenize(url)[:42] + ['[SEP]'] + self.tokenizer.tokenize(title)[:41] + ['[SEP]']
        body_tokens = self.tokenizer.tokenize(body)[:(self.max_seq_length - len(prev_tokens) - 1)]
        passage = prev_tokens + body_tokens + ['[SEP]']

        passage = self.tokenizer.convert_tokens_to_ids(passage)[:self.max_seq_length]
        
        return real_index, passage

    @classmethod
    def get_collate_fn(cls, pad_token_id):
        def create_passage_input(features):
            index_list = [int(x[0]) for x in features]
            d_list = [x[1] for x in features]
            max_d_len = max([len(d) for d in d_list])
            d_list = [d + [pad_token_id] * (max_d_len - len(d)) for d in d_list]
            doc_tensor = torch.LongTensor(d_list)
            return index_list, doc_tensor, (doc_tensor != 0).long()
        return create_passage_input

def inference(model, dataloader, is_query=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()
    if device == torch.device("cuda") and gpu_num > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    model.to(device)
    model.eval()

    id_list, embeddings_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if is_query:
                text_id, encode_passages = batch
                encode_passages = to_device(encode_passages, device)
            else:
                text_id, text_ids, text_mask = batch
                encode_passages = {"input_ids": to_device(text_ids, device), "attention_mask": to_device(text_mask, device)}
            outputs = model(**encode_passages)
            
            # sent_emb = outputs.last_hidden_state[:, 0]
            sent_emb = outputs["sent_emb"]
            sent_emb = sent_emb.cpu().numpy()
            id_list.extend(text_id)
            embeddings_list.extend(sent_emb)
    
    id_list, embeddings_list = np.array(id_list, dtype=np.int32), np.array(embeddings_list, dtype=np.float32)
    id_list = np.reshape(id_list, (-1, 1))
    return id_list, embeddings_list

def store_embeddings(id_list, embeddings_list, output_path, name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_fbin(os.path.join(output_path, "{}_embeddings.fbin".format(name)), embeddings_list)
    write_ibin(os.path.join(output_path, "{}_ids.ibin".format(name)), id_list)

def load_embeddings(embedding_dir, name):
    embeddings = read_fbin(os.path.join(embedding_dir, "{}_embeddings.fbin".format(name)))
    id_list = read_ibin(os.path.join(embedding_dir, "{}_ids.ibin".format(name)))
    return embeddings, id_list

def gpu_retrieval(query_embeddings, passage_embeddings, topk):
    faiss.omp_set_num_threads(90)
    cpu_index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index,
        co=co
    )
    gpu_index_flat.add(passage_embeddings)
    scores, indices = gpu_index_flat.search(query_embeddings, topk)
    return scores, indices

def cpu_retrieval(query_embeddings, passage_embeddings, topk):
    cpu_index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    cpu_index.add(passage_embeddings)
    scores, indices = cpu_index.search(query_embeddings, topk)
    return scores, indices

def eval_dense():
    parser = parser = HfArgumentParser((EvaluationConfig, DataConfig, ModelConfig))

    eval_config, data_config, model_config = parser.parse_args_into_dataclasses()
    eval_config: EvaluationConfig
    data_config: DataConfig
    model_config: ModelConfig
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if eval_config.search:
        logger.info("Skip loading model.")
    else:
        logger.info("MODEL parameters %s", model_config)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_name,
            cache_dir=model_config.model_cache_dir 
        )
        model_config.vocab_size = len(tokenizer)
        eval_config.vocab_size = len(tokenizer)
        model_config.half = eval_config.fp16
        full_model = BiEncoder(model_config)
        logger.info("Model loaded.")
        
    if eval_config.encode_query:
        logger.info("--------------------- ENCODE QUERY PROCEDURE ---------------------")
        is_query = True
        query_lmdb_env = lmdb.open(data_config.query_lmdb_dir, subdir=os.path.isdir(data_config.query_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        with query_lmdb_env.begin(write=False) as txn:
            n_query = pickle.loads(txn.get(b'__len__'))

        query_dataset = MARCOWSTestIdsDataset(query_lmdb_env, tokenizer,
                                               start_idx=0, end_idx=n_query,
                                               max_length=data_config.q_max_len)
        
        query_loader = DataLoader(
            query_dataset,
            batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
            collate_fn=PredictionCollator(
                tokenizer=tokenizer,
                max_length=data_config.q_max_len,
                is_query=is_query
            ),
            num_workers=eval_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        model = full_model.q_encoder

        id_list, embeddings_list = inference(model, query_loader, is_query)
        logger.info(f"Query embeddings shape: {embeddings_list.shape}")
        
        store_embeddings(id_list, embeddings_list, eval_config.embedding_output_dir, "query")
        
    if eval_config.encode_corpus:
        logger.info("--------------------- ENCODE CORPUS PROCEDURE ---------------------")
        is_query = False
        model = full_model.k_encoder

        corpus_lmdb_env = lmdb.open(data_config.passage_lmdb_dir, subdir=os.path.isdir(data_config.passage_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        with corpus_lmdb_env.begin(write=False) as txn:
            n_corpus = pickle.loads(txn.get(b'__len__'))
        id_mapper = json.load(open(data_config.idmapping_path))

        shards_num = eval_config.shards_num
        assert shards_num > 0
        shard_size = n_corpus // shards_num        
        
        for shard_id in range(shards_num):
            start_idx = shard_id * shard_size
            end_idx = start_idx + shard_size
            if shard_id < eval_config.start_shard:
                logger.info("Skip shard: {}".format(shard_id))
                continue
            if shard_id == shards_num - 1:
                end_idx = n_corpus
            logger.info(f"Inference shard {shard_id} from {start_idx} to {end_idx}, num passages: {end_idx - start_idx}")

            # corpus_dataset = MARCOWSTestIdsDataset(corpus_lmdb_env, tokenizer,
            #                                     start_idx=start_idx, end_idx=end_idx,
            #                                     idmapping=id_mapper,
            #                                     max_length=data_config.k_max_len)
            # corpus_loader = DataLoader(
            #     corpus_dataset,
            #     batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
            #     collate_fn=PredictionCollator(
            #         tokenizer=tokenizer,
            #         max_length=data_config.k_max_len,
            #         is_query=is_query
            #     ),
            #     num_workers=eval_config.dataloader_num_workers,
            #     pin_memory=True,
            #     persistent_workers=True
            # )
            corpus_dataset = TextDatasetLMDBMeta(start_idx=start_idx, end_idx=end_idx,
                                                 doc_pool_txn=corpus_lmdb_env.begin(write=False), tokenizer=tokenizer,
                                                 id2id=id_mapper)
            corpus_loader = DataLoader(corpus_dataset, batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
                drop_last=False,
                collate_fn=TextDatasetLMDBMeta.get_collate_fn(tokenizer.pad_token_id),
                pin_memory=True, persistent_workers=True,
                num_workers=eval_config.dataloader_num_workers,
            )
            id_list, embeddings_list = inference(model, corpus_loader)
            logger.info(f"Corpus embeddings shape: {embeddings_list.shape}")
            store_embeddings(id_list, embeddings_list, eval_config.embedding_output_dir, "corpus_shard{:02d}".format(shard_id))

    if eval_config.search:
        logger.info("--------------------- SEARCH PROCEDURE ---------------------")
        # store_embeddings(id_list, embeddings_list, eval_config.embedding_output_dir, "corpus_shard{:02d}".format(shard_id))
        query_embeddings, query_ids = load_embeddings(eval_config.embedding_dir, "query")
        query_ids = query_ids.reshape(-1)
        logger.info(f"Query embeddings shape: {query_embeddings.shape}")
        
        id_mapper = json.load(open(data_config.idmapping_path))
        shards_num = eval_config.shards_num
        if eval_config.use_gpu:
            retrieval_func = gpu_retrieval
        else:
            retrieval_func = cpu_retrieval
        score_list, index_list = [], []
        for shard_id in range(shards_num):
            shard_embeddings, shard_ids = load_embeddings(eval_config.embedding_dir, "corpus_shard{:02d}".format(shard_id))
            shard_ids = shard_ids.reshape(-1)
            logger.info(f"Search in shard {shard_id}, embeddings length: {shard_embeddings.shape[0]}")
            
            scores, indices = retrieval_func(query_embeddings, shard_embeddings, topk=eval_config.retrieve_topk)
            real_dis = shard_ids[indices]
            
            score_list.append(scores)
            index_list.append(real_dis)
        logger.info("Merging results...")
        score_list = np.concatenate(score_list, axis=0)
        index_list = np.concatenate(index_list, axis=0)
        arg_indices = np.argsort(-score_list, axis=1)
        final_score_list = np.take_along_axis(score_list, arg_indices, axis=1)
        final_index_list = np.take_along_axis(index_list, arg_indices, axis=1)

        res_full = {}
        for qid, scores, indices in zip(query_ids, final_score_list, final_index_list):
            res_full[qid] = [indices, scores]

        if eval_config.eval_gt_path:
            eval_result = compute_metrics(load_gt(eval_config.eval_gt_path), res_full)
            print(eval_result)
        else:
            os.makedirs(eval_config.retrieve_result_output_dir, exist_ok=True)
            save_ranking_path = os.path.join(eval_config.retrieve_result_output_dir, "flatip_ranking.pkl")
            logger.info("No qrels file, save result to: {}".format(save_ranking_path))
            with open(save_ranking_path, 'wb') as f:
                pickle.dump(res_full, f)


if __name__ == "__main__":
    eval_dense()