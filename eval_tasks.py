import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba.typed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pickle
import numba
import json
import numpy as np
import logging
from collections import defaultdict

import logging



from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)


logger = logging.getLogger(__name__)

from tqdm import tqdm
from inverted_index import InvertedIndex
from auxiliary import (
    to_device,
    DataConfig, 
    ModelConfig, 
    EvaluationConfig
)
logger = logging.getLogger(__name__)

class Evalutator:
    def __init__(self, model, config=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_num = torch.cuda.device_count()
        print("GPU count: {}".format(self.gpu_num))
        model.to(self.device)
        
        if self.device == torch.device("cuda") and self.gpu_num > 1:
            model = nn.DataParallel(model, device_ids=[i for i in range(self.gpu_num)])
        model.eval()
        self.model = model

class SparseIndex(Evalutator):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.index_dir = config.index_dir if config is not None else None
        self.index_filename = config.index_filename
        self.invert_index = InvertedIndex(self.index_dir, config.index_filename, voc_dim=config.vocab_size, force_new=config.force_build_index)
        self.kterm_num = config.kterm_num

    def index(self, corpus_loader):
        doc_ids = []
        row_count = 0
        with torch.no_grad():
            for batch in tqdm(corpus_loader):
                
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

                row_count += outputs.size(0)
                doc_ids.extend(text_id)
                self.invert_index.add_batch_document(rows, 
                                                     doc_dim.view(-1).cpu().numpy(), 
                                                     values.view(-1).cpu().numpy(), 
                                                     len(text_id))
    

        if self.index_dir is not None:
            self.invert_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids_{}.pkl".format(self.index_filename)), "wb"))
            logger.info("Index saved at {}".format(self.index_dir))


class SparseRetriever(Evalutator):
    def __init__(self, model, config):
        super().__init__(model, config)

        self.output_dir = config.retrieve_result_output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.qterm_num = config.qterm_num
        
        if config.do_query_encode:
            return
        
        self.invert_index = InvertedIndex(config.index_dir, config.index_filename, voc_dim=config.vocab_size)
        self.doc_ids = pickle.load(open(os.path.join(config.index_dir, "doc_ids_{}.pkl".format(config.index_filename)), "rb"))
        
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()

        for k, v in self.invert_index.index_doc_id.items():
            self.numba_index_doc_ids[k] = v
        for k, v in self.invert_index.index_doc_value.items():
            self.numba_index_doc_values[k] = v
        
    def retrieve_from_json(self, query_path, top_k=200, threshold=0):
        res = defaultdict(dict)
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
                for idx, sc in zip(indices, scores):
                    res[str(query["text_id"])][str(self.doc_ids[idx])] = float(sc)   
        return res
    
    def retrieve(self, query_loader, top_k=200, threshold=0):
        res = defaultdict(dict)
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
                for idx, sc in zip(indices, scores):
                    res[str(query_id[0])][str(self.doc_ids[idx])] = float(sc)   
        with open(os.path.join(self.output_dir, "result.json"), "w") as f:
            json.dump(res, f)
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
                # json_list.append({"text_id": int(query_id[0]), "text": doc_dim.tolist(), "value": values.tolist()})
                f.write(json.dumps({"text_id": int(query_id[0]), "text": doc_dim.tolist(), "value": values.tolist()}) + "\n")            
                
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
            retrieved_indice = invert_index_ids[query_index]
            retrieved_values = invert_index_values[query_index]
            for j in numba.prange(len(retrieved_indice)):
                scores[retrieved_indice[j]] += query_value * retrieved_values[j]
        filtered_indices = np.argwhere(scores > threshold)[:, 0]
        return filtered_indices, -scores[filtered_indices]

    @staticmethod
    def select_topk(indices, scores, k):
        if len(indices) > k:
            parted_idx = np.argpartition(scores, k)[: k]
            indices, scores = indices[parted_idx], scores[parted_idx]

        sorted_idx = np.argsort(scores)
        sorted_indices, sorted_scores = indices[sorted_idx], scores[sorted_idx]
        
        return sorted_indices, -sorted_scores


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
        model_config.tokenizer_name if model_config.tokenizer_name else model_config.model_name_or_path,
        cache_dir=model_config.model_cache_dir 
    )
    model_config.vocab_size = len(tokenizer)
    eval_config.vocab_size = len(tokenizer)
    model_config.half = eval_config.fp16

    # if model_config.sparse_shared_encoder:
    #     model = SparseSharedEncoder.build_model(model_config)
    # else:
    model = None
    
    logger.info("Model loaded.")

    if eval_config.do_corpus_index:
        logger.info("--------------------- CORPUS INDEX PROCEDURE ---------------------")
        is_query = False
        corpus_dataset = PredictionDataset(data_config, tokenizer, is_query=is_query)
        
        corpus_loader = DataLoader(
            corpus_dataset,
            batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
            collate_fn=PredictionCollator(
                tokenizer=tokenizer,
                max_length=data_config.k_max_len,
                is_query=is_query
            ),
            num_workers=eval_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        indexer = SparseIndex(
            model,
            eval_config
        )
        indexer.index(corpus_loader)
        
    if eval_config.do_retrieve:
        logger.info("--------------------- RETRIEVAL PROCEDURE ---------------------")
        is_query = True

        query_dataset = PredictionDataset(data_config, tokenizer, is_query=is_query)
        
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

        retriever = SparseRetriever(
            model,
            eval_config
        )
        res = retriever.retrieve(query_loader, top_k=eval_config.retrieve_topk)
        if eval_config.save_ranking:
            save_name = eval_config.save_name if eval_config.save_name is not None else "splade_rank.pkl" 
            with open(os.path.join(eval_config.retrieve_result_output_dir, save_name), 'wb') as f:
                pickle.dump(res, f)
       
    if eval_config.do_query_encode:
        logger.info("--------------------- QUERY ENCODE PROCEDURE ---------------------")
        is_query = True

        query_dataset = PredictionDataset(data_config, tokenizer, is_query=is_query)
        
        query_loader = DataLoader(
            query_dataset,
            batch_size=1, #eval_config.per_device_eval_batch_size * torch.cuda.device_count(), # just one at a time for now
            collate_fn=PredictionCollator(
                tokenizer=tokenizer,
                max_length=data_config.q_max_len,
                is_query=is_query
            ),
            num_workers=eval_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        retriever = SparseRetriever(
            model,
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
        retriever = SparseRetriever(
            model,
            eval_config
        )
        res = retriever.retrieve_from_json(eval_config.query_json_path, top_k=eval_config.retrieve_topk)


if __name__ == "__main__":
    splade_eval()