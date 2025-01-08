import os
import sys
import lmdb
import json
import torch
import torch.nn as nn
import logging
import pickle
import struct
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
from eval_splade import SparseRetriever, SparseIndex, PredictionCollator, MARCOWSTestIdsDataset
from eval_dense import inference, store_embeddings

logger = logging.getLogger(__name__)


class TextSubSetDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self, doc_pool_txn, tokenizer, max_length, sampled_ids):
        self.doc_pool_txn = doc_pool_txn
        self.length = len(sampled_ids)
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.sampled_ids = sampled_ids

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        real_index = str(self.sampled_ids[index])
        url, title, body = pickle.loads(self.doc_pool_txn.get(real_index.encode()))

        prev_tokens = ['[CLS]']  + self.tokenizer.tokenize(url)[:42] + ['[SEP]'] + self.tokenizer.tokenize(title)[:41] + ['[SEP]']
        body_tokens = self.tokenizer.tokenize(body)[:(self.max_seq_length - len(prev_tokens) - 1)]
        passage = prev_tokens + body_tokens + ['[SEP]']
        passage = self.tokenizer.convert_tokens_to_ids(passage)[:self.max_seq_length]
        return real_index, passage

    @classmethod
    def get_collate_fn(cls, args):
        def create_passage_input(features):
            index_list = [x[0] for x in features]
            d_list = [x[1] for x in features]
            max_d_len = max([len(d) for d in d_list])
            d_list = [d + [args.pad_token_id] * (max_d_len - len(d)) for d in d_list]
            doc_tensor = torch.LongTensor(d_list)
            return index_list, doc_tensor, (doc_tensor != 0).long()
        return create_passage_input


def eval_subset_splade():
    parser = HfArgumentParser((EvaluationConfig, DataConfig, ModelConfig))

    eval_config, data_config, model_config = parser.parse_args_into_dataclasses()
    eval_config: EvaluationConfig
    data_config: DataConfig
    model_config: ModelConfig
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

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

    if eval_config.do_corpus_index:
        
        is_query = False
        logger.info("--------------------- CORPUS INDEX PROCEDURE ---------------------")
        passage_lmdb_env = lmdb.open(data_config.passage_lmdb_dir, subdir=os.path.isdir(data_config.passage_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        
        with open(eval_config.sampled_docid_path, "rb") as f:
            sampled_ids = pickle.load(f)
        doc_pool_txn = passage_lmdb_env.begin(write=False)
        corpus_dataset = TextSubSetDatasetLMDB(doc_pool_txn, tokenizer, 
                                            sampled_ids=sampled_ids,
                                            max_length=data_config.k_max_len)
        
        logger.info(f"Dataset size: {len(corpus_dataset)}")
        corpus_loader = DataLoader(corpus_dataset, batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
                drop_last=False,
                collate_fn=TextSubSetDatasetLMDB.get_collate_fn(tokenizer.pad_token_id),
                pin_memory=True, persistent_workers=True,
                num_workers=eval_config.dataloader_num_workers,
        )
        
        model = full_model.k_encoder
        indexer = SparseIndex(
            model,
            os.path.join(eval_config.index_dir, "subset_splade_index"),
            eval_config
        )
        indexer.index(corpus_loader)
        
    if eval_config.do_retrieve:
        logger.info("--------------------- RETRIEVAL PROCEDURE ---------------------")
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
            batch_size=1, # just one at a time for now
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
        shards_num = eval_config.shards_num
        assert shards_num > 0
        model = full_model.q_encoder
        
        res_full = {}
        for shard_id in range(shards_num):
            logger.info(f"Searching in shard {shard_id}")
            retriever = SparseRetriever(
                model,
                os.path.join(eval_config.index_dir, f"shard_{shard_id}"),
                eval_config
            )
            res = retriever.retrieve(query_loader, top_k=eval_config.retrieve_topk)
            for k, v in res.items():
                if shard_id == 0:
                    res_full[k] = v
                else:
                    res_full[k][0] = np.concatenate((res_full[k][0], v[0]), axis=0) # Indices
                    res_full[k][1] = np.concatenate((res_full[k][1], v[1]), axis=0) # Scores

        top_k = eval_config.retrieve_topk
        for k, v in tqdm(res_full.items(), total=len(res_full), desc="Select topk"):
            indices, scores = v
            indices, scores = retriever.select_topk(indices, scores, k=top_k)
            res_full[int(k)] = indices, scores
        
        

        if eval_config.eval_gt_path:
            eval_result = compute_metrics(load_gt(eval_config.eval_gt_path), res_full)
            print(eval_result)
        else:
            os.makedirs(eval_config.retrieve_result_output_dir, exist_ok=True)
            save_ranking_path = os.path.join(eval_config.retrieve_result_output_dir, "splade_ranking.pkl")
            logger.info("No qrels file, save result to: {}".format(save_ranking_path))
            with open(save_ranking_path, 'wb') as f:
                pickle.dump(res_full, f)
    
    if eval_config.do_query_encode:
        logger.info("--------------------- QUERY ENCODE PROCEDURE ---------------------")
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
            batch_size=1, # eval_config.per_device_eval_batch_size * torch.cuda.device_count(), # just one at a time for now
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
        retriever = SparseRetriever(
            model,
            eval_config.index_dir,
            eval_config
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_num = torch.cuda.device_count()
        if device == torch.device("cuda") and gpu_num > 1:
            model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
        model.to(device)
        model.eval()
        json_list = retriever.save_encode(model, query_loader, os.path.join(eval_config.index_dir, eval_config.save_name))

        with open(os.path.join(eval_config.index_dir, "spalde_query.bin"), "wb") as fw:
            total_query = len(json_list)
            print("total query:", total_query)
            fw.write(struct.pack("I", total_query))
            for line in tqdm(json_list, total=total_query, desc="Saving to bins"):
                content = json.loads(line.strip())
                query_ids = np.array(content["text"], dtype=np.int32)
                query_value = np.array(content["value"] if "value" in content else [1.0 for _ in range(len(query_ids))], dtype=np.float32)
                fw.write(struct.pack("I", len(query_ids)))
                ids_size, values_size = query_ids.nbytes, query_value.nbytes
                fw.write(struct.pack("I", ids_size))
                fw.write(query_ids.tobytes())
                fw.write(struct.pack("I", values_size))
                fw.write(query_value.tobytes())


def eval_subset_dense():
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
        with open(eval_config.sampled_docid_path, "rb") as f:
            sampled_ids = pickle.load(f)
            
        doc_pool_txn = corpus_lmdb_env.begin(write=False)
        corpus_dataset = TextSubSetDatasetLMDB(doc_pool_txn, tokenizer, 
                                            sampled_ids=sampled_ids,
                                            max_length=data_config.k_max_len)
        
        logger.info(f"Dataset size: {len(corpus_dataset)}")
        corpus_loader = DataLoader(corpus_dataset, batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
                drop_last=False,
                collate_fn=TextSubSetDatasetLMDB.get_collate_fn(tokenizer.pad_token_id),
                pin_memory=True, persistent_workers=True,
                num_workers=eval_config.dataloader_num_workers,
        )
        
        id_list, embeddings_list = inference(model, corpus_loader)
        logger.info(f"Corpus embeddings shape: {embeddings_list.shape}")
        store_embeddings(id_list, embeddings_list, eval_config.embedding_output_dir, "corpus_subset")

    
if __name__ == "__main__":
    eval_subset_dense()
    eval_subset_splade()