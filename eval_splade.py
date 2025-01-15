import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lmdb
import numba.typed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import pickle
import numba
import json
import numpy as np
import logging
from collections import defaultdict

from dataclasses import dataclass
import logging



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
    EvaluationConfig
)
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


class Evalutator:
    def __init__(self, model, config=None):
        self.config = config
        if model is None:
            self.model = None
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_num = torch.cuda.device_count()
        logger.info("GPU count: {}".format(self.gpu_num))
        model.to(self.device)
        
        if self.device == torch.device("cuda") and self.gpu_num > 1:
            model = nn.DataParallel(model, device_ids=[i for i in range(self.gpu_num)])
        model.eval()
        self.model = model


class SparseIndex(Evalutator):
    def __init__(self, model, index_dir, config):
        super().__init__(model, config)
        self.index_dir = index_dir
        self.index_filename = config.index_filename
        self.invert_index = InvertedIndex(self.index_dir, config.index_filename, force_rebuild=config.force_build_index)
        self.kterm_num = config.kterm_num

    def index(self, corpus_loader):
        doc_ids = []
        row_count = 0
        assert self.index_dir is not None
        with torch.no_grad():
            for batch in tqdm(corpus_loader):
                # text_id, text_ids, text_mask = batch
                # encode_passages = {"input_ids": to_device(text_ids, self.device), "attention_mask": to_device(text_mask, self.device)}
                
                text_id, encode_passages = batch
                encode_passages = to_device(encode_passages, self.device)
                outputs = self.model(**encode_passages)
        
                outputs = outputs["vocab_reps"]
                if self.kterm_num is not None:
                    values, doc_dim = torch.topk(outputs, self.kterm_num, dim=1)
                    rows = np.repeat(np.arange(row_count, row_count + outputs.size(0)), self.kterm_num)
                else:
                    rows, doc_dim = torch.nonzero(outputs, as_tuple=True)
                    values = outputs[rows.detach().cpu().tolist(), doc_dim.detach().cpu().tolist()]
                    rows = rows.detach().cpu() + row_count
                # print(values.shape)
                row_count += outputs.size(0)
                doc_ids.extend(text_id)
                self.invert_index.add_batch_item(rows, doc_dim.view(-1).cpu().numpy(), values.view(-1).cpu().numpy())
    

        self.invert_index.save()
        pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids_{}.pkl".format(self.index_filename)), "wb"))
        logger.info("Index saved at {}".format(self.index_dir))


class SparseRetriever(Evalutator):
    def __init__(self, model, index_dir, config):
        super().__init__(model, config)

        self.output_dir = config.retrieve_result_output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.qterm_num = config.qterm_num
        
        if config.do_query_encode:
            return
        
        self.invert_index = InvertedIndex(index_dir, config.index_filename)
        self.doc_ids = pickle.load(open(os.path.join(index_dir, "doc_ids_{}.pkl".format(config.index_filename)), "rb"))
        
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()

        for k, v in self.invert_index.index_ids.items():
            self.numba_index_doc_ids[k] = v
        for k, v in self.invert_index.index_values.items():
            self.numba_index_doc_values[k] = v
        # 5087696
        # 20214074
    def retrieve_from_json(self, query_path, top_k=100, threshold=0):
        res = {}
        with open(query_path, "r") as f:
            for line in tqdm(f):
                query = json.loads(line.strip())
                indices, scores = self.sparse_match(self.numba_index_doc_ids,
                                                    self.numba_index_doc_values,
                                                    np.array(query["text"]),
                                                    np.array(query["value"]),
                                                    threshold,
                                                    self.invert_index.total_docs)
                indices, scores = self.select_topk(indices, scores, k=top_k)
                # print(indices)
                indices = np.array([int(self.doc_ids[i]) for i in indices])
                # print(indices)
                res[int(query["text_id"])] = [indices, scores]  
                # break
        return res
    
    def retrieve(self, query_loader, top_k=100, threshold=0):
        res = {}
        with torch.no_grad():
            for batch in tqdm(query_loader):

                query_id, encode_query = batch
                encode_query = to_device(encode_query, self.device)
                outputs = self.model(**encode_query)
                outputs = outputs["vocab_reps"]
                if self.qterm_num is not None:
                    values, doc_dim = torch.topk(outputs, self.qterm_num, dim=1)
                else:
                    row, doc_dim = torch.nonzero(outputs, as_tuple=True)
                    values = outputs[row.detach().cpu().tolist(), doc_dim.detach().cpu().tolist()]
                indices, scores = self.sparse_match(self.numba_index_doc_ids,
                                                    self.numba_index_doc_values,
                                                    doc_dim.view(-1).cpu().numpy(),
                                                    values.view(-1).cpu().numpy(),
                                                    threshold,
                                                    self.invert_index.total_docs)
                indices, scores = self.select_topk(indices, scores, k=top_k)    
                # for idx, sc in zip(indices, scores):
                #     res[str(query_id[0])][str(self.doc_ids[idx])] = float(sc)   
                res[int(query_id[0])] = [indices, scores]
        # with open(os.path.join(self.output_dir, "result.json"), "w") as f:
        #     json.dump(res, f)
        return res

    def save_encode(self, model, query_loader, output_path):
        res = defaultdict(dict)
        json_list = []
        with torch.no_grad() and open(output_path, "w") as f:
            for batch in tqdm(query_loader):

                query_id, encode_query = batch
                encode_query = to_device(encode_query, self.device)
                outputs = model(**encode_query)
                outputs = outputs["vocab_reps"]
                if self.qterm_num is not None:
                    values, doc_dim = torch.topk(outputs, self.qterm_num, dim=1)
                else:
                    row, doc_dim = torch.nonzero(outputs, as_tuple=True)
                    values = outputs[row.detach().cpu().tolist(), doc_dim.detach().cpu().tolist()]
                doc_dim, values = doc_dim.view(-1).detach().cpu().numpy(), values.view(-1).detach().cpu().numpy()
                f.write(json.dumps({"text_id": int(query_id[0]), "text": doc_dim.tolist(), "value": values.tolist()}) + "\n")
        return json_list            
                
    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def sparse_match(invert_index_ids: numba.typed.Dict,
                          invert_index_values: numba.typed.Dict,
                          query_indices: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          corpus_size: int):
        scores = np.zeros(corpus_size, dtype=np.float32)
        N = len(query_indices)
        for i in range(N):
            query_index, query_value = query_indices[i], query_values[i]
            try:
                retrieved_indice = invert_index_ids[query_index]
                retrieved_values = invert_index_values[query_index]
            except:
                continue
            for j in numba.prange(len(retrieved_indice)):
                scores[retrieved_indice[j]] += query_value * retrieved_values[j]
        filtered_indices = np.argwhere(scores > threshold)[:, 0]
        return filtered_indices, -scores[filtered_indices]

    @staticmethod
    def select_topk(indices, scores, k):
        if len(indices) > k:
            parted_idx = np.argpartition(scores, k)[: k]
            indices, scores = indices[parted_idx], scores[parted_idx]
        # scores = -scores
        sorted_idx = np.argsort(scores)
        sorted_indices, sorted_scores = indices[sorted_idx], scores[sorted_idx]
        
        return sorted_indices, sorted_scores


def search_in_shard(shard_id, eval_config):
    logger.info(f"Searching in shard {shard_id}")
    retriever = SparseRetriever(
        None,
        os.path.join(eval_config.index_dir, f"shard_{shard_id}") ,
        eval_config
    )
    res = retriever.retrieve_from_json(eval_config.query_json_path, top_k=eval_config.retrieve_topk)     
    return shard_id, res

def splade_eval():
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

    # if model_config.sparse_shared_encoder:
    #     model = SparseSharedEncoder.build_model(model_config)
    # else:
    full_model = BiEncoder(model_config)
    
    
    logger.info("Model loaded.")

    if eval_config.do_corpus_index:
        
        is_query = False
        logger.info("--------------------- CORPUS INDEX PROCEDURE ---------------------")
        passage_lmdb_env = lmdb.open(data_config.passage_lmdb_dir, subdir=os.path.isdir(data_config.passage_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        id_mapper = json.load(open(data_config.idmapping_path))

        with passage_lmdb_env.begin(write=False) as txn:
            n_passages = pickle.loads(txn.get(b'__len__'))
        shards_num = eval_config.shards_num
        assert shards_num > 0
        shard_size = n_passages // shards_num
        
        for shard_id in range(shards_num):
            start_idx = shard_id * shard_size
            end_idx = start_idx + shard_size
            if shard_id < eval_config.start_shard:
                logger.info("Skip shard: {}".format(shard_id))
                continue
            if shard_id == shards_num - 1:
                end_idx = n_passages
            logger.info(f"Indexing shard {shard_id} from {start_idx} to {end_idx}, num passages: {end_idx - start_idx}")

            corpus_dataset = MARCOWSTestIdsDataset(passage_lmdb_env, tokenizer, 
                                                start_idx=start_idx, end_idx=end_idx,
                                                idmapping=id_mapper,
                                                max_length=data_config.k_max_len)
            logger.info(f"Dataset size: {len(corpus_dataset)}")
            corpus_loader = DataLoader(
                corpus_dataset,
                batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
                collate_fn=PredictionCollator(
                    tokenizer=tokenizer,
                    max_length=data_config.k_max_len,
                ),
                num_workers=eval_config.dataloader_num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            model = full_model.k_encoder
            indexer = SparseIndex(
                model,
                os.path.join(eval_config.index_dir, f"shard_{shard_id}"),
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
        res = retriever.save_encode(model, query_loader, os.path.join(eval_config.index_dir, eval_config.save_name))

    if eval_config.do_retrieve_from_json: 
        logger.info("--------------------- RETRIEVAL FROM JSON PROCEDURE ---------------------")
        shards_num = eval_config.shards_num
        start_shard = eval_config.start_shard
        assert shards_num > 0
        model = full_model.q_encoder
        
        res_full = {}
        pool = mp.Pool(processes=shards_num)
        results = []
        for shard_id in range(start_shard, start_shard+shards_num):
            results.append(pool.apply_async(search_in_shard, args=(shard_id, eval_config)))
        
        pool.close()
        pool.join()
        for result in results:
            shard_id, res = result.get()
            for k, v in res.items():
                if k not in res_full:
                    res_full[k] = v
                else:
                    res_full[k][0] = np.concatenate((res_full[k][0], v[0]), axis=0) # Indices
                    res_full[k][1] = np.concatenate((res_full[k][1], v[1]), axis=0) # Scores

        top_k = eval_config.retrieve_topk
        for k, v in tqdm(res_full.items(), total=len(res_full), desc="Select topk"):
            indices, scores = v
            indices, scores = SparseRetriever.select_topk(indices, scores, k=top_k)
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


if __name__ == "__main__":
    splade_eval()