import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import argparse
import logging
import pickle
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import SPTAG
from collections import defaultdict
from numba import njit
from numba.typed import Dict, List
from numba import types
import multiprocessing

from inverted_index import InvertedIndex, BM25Retriever
from auxiliary import read_fbin, load_gt, compute_metrics


def build_invert_index(args):
    index = InvertedIndex(args.spann_index_dir, args.spann_index_name, force_rebuild=args.force_rebuild)
    with open(args.cluster_file, "r") as f:
        for cid, line in tqdm(enumerate(f.readlines())):
            cid = int(cid)
            doc_ids = line.split(",")
            for did in doc_ids:
                index.add_item(int(did), cid, 1.0)
    index.save()

def sptag_search(index, embedding, cluster_num=256):
    """
    return a tuple(cluster_id, score), latency
    """
    start = timer()
    # top 256 clusters
    result = index.Search(embedding, cluster_num) # the search results are not docid, but cluster id
    latency = timer() - start
    return [(result[0][i], float(result[1][i])) for i in range(len(result[1]))], latency

def reciprocal_dis(query_embedding, cand_doc_embd):
    return 1 / (1e-6 + np.sum((query_embedding - cand_doc_embd) ** 2, axis=1))

def opposite_dis(query_embedding, cand_doc_embd):
    return - np.sum((query_embedding - cand_doc_embd) ** 2, axis=1)

def inner_product(query_embedding, cand_doc_embd):
    return np.sum(query_embedding.reshape(1, -1) * cand_doc_embd, axis=1)
@njit
def add_scores(cand, postings, scores, values):
    for i in range(len(postings)):
        for j in range(len(postings[i])):
            docid = postings[i][j]
            score = scores[i][j] * values[i]
            if docid in cand:
                cand[docid] += score
            else:
                cand[docid] = score

def prepare_query(query_text_path, query_emb_path, qlookup_path):
    with open(query_text_path, "r") as f:
        query_texts = [json.loads(line.strip()) for line in f]
    query_embeddings = read_fbin(query_emb_path)
    with open(qlookup_path, "rb") as f:
        qid = pickle.load(f)
    return query_texts, query_embeddings, qid

class Searcher:
    def __init__(self, 
                 splade_index, spann_index,  
                 sptag_index, doc_embedding=None,
                 this_weight=1.0, that_weight=10000.0,
                 use_cmp=False, emb_dis_method="rep"):
        self.splade_index = splade_index
        self.spann_index = spann_index
        self.sptag_index = sptag_index
        self.splade_index.engage_numba()

        self.this_weight = this_weight
        self.that_weight = that_weight

        self.doc_embedding = doc_embedding
        self.use_cmp = use_cmp

        # if emb_dis_method == "rep":
        #     self.dis_fn = reciprocal_dis
        # elif emb_dis_method == "opp":
        #     self.dis_fn = opposite_dis
        # else:
        #     raise NotImplementedError
        self.dis_fn = inner_product
    
    def search(self, query, query_embedding, topk=100, cluster_num=256):
        query_text = query["text"]
        query_value = query["value"] if "value" in query else [1.0 for _ in range(len(query_text))]
        
        cluster_list, time1 = sptag_search(self.sptag_index, np.array(query_embedding.tolist(), dtype=np.float32), cluster_num)
        cid_list, cid_score = [cluster[0] for cluster in cluster_list], [ cluster[1] for cluster in cluster_list]
        
        start = timer()
        
        splade_postings, splade_values = self.splade_index.get_postings(query_text)
        
        t1 = timer()
        
        cand1 = Dict.empty(
            key_type=types.int64,
            value_type=types.float64,
        )
        add_scores(cand1, splade_postings, splade_values, query_value)

        t11 = timer() 
        
        spann_postings, spann_values = self.spann_index.get_postings(cid_list)
        
        t21 = timer()
        
        if self.use_cmp:
            cand2 = defaultdict(float)
            for posting, score in zip(spann_postings, cid_score):
                for docid in posting:
                        cand2[docid] = score 
        else:    
            cand2_key_list = []    
            for posting in spann_postings:
                for docid in posting:
                    cand2_key_list.append(docid)
            cand_doc_embd = self.doc_embedding[cand2_key_list]
            cand2_dis = self.dis_fn(query_embedding, cand_doc_embd) 
            cand2 = dict(zip(cand2_key_list, cand2_dis))

        t2 = timer()
        
        cand = {k:  cand1.get(k, 0) * self.this_weight + cand2.get(k, 0) * self.that_weight
                 for k in set(cand1.keys()) & set(cand2.keys())} 

        cand = dict(sorted(cand.items(), key=lambda x: x[1], reverse=True))
        docs, scores = np.array(list(cand.keys())[:topk]), np.array(list(cand.values())[:topk])

        end = timer()
        
        time_dict = {"total": end - start, 
                     "sptag": time1, 
                     "retrieve_postings1": t1 - start, 
                     "merge_postings1": t11 - t1,
                     "retrieve_postings2": t21 - t11,
                     "merge_postings2": t2 - t21,
                     "get_topk": end - t2}

        return docs, scores, time_dict
    
    @classmethod
    def build(cls, 
              splade_index_dir, splade_index_name,
              spann_index_dir, spann_index_name, 
              sptags_index_path, doc_emb_path,
              this_weight, that_weight, 
              use_cmp, use_v2, emb_dis_method):
        sptag_index = SPTAG.AnnIndex.Load(sptags_index_path)
        if not use_v2:
            splade_index = BM25Retriever(splade_index_dir, splade_index_name)
        else:
            splade_index = InvertedIndex(splade_index_dir, splade_index_name) # temp magic number

        spann_index = InvertedIndex(spann_index_dir, spann_index_name)
        if doc_emb_path is not None:
            doc_embedding = read_fbin(doc_emb_path)
        else:
            doc_embedding = None

        return cls(splade_index, spann_index, 
                   sptag_index, doc_embedding, 
                   this_weight, that_weight, 
                   use_cmp, emb_dis_method)



def process_queries(queries, query_embeddings, qid2idx, id_mapper, doer, args):
    result = {}
    local_time_dict = defaultdict(float)

    for query_text in tqdm(queries):
        qid = int(query_text["text_id"])
        query_embedding = query_embeddings[qid2idx[qid]]
        indice, scores, time_dict = doer.search(query_text, query_embedding, 
                                                topk=args.depth,
                                                cluster_num=args.cluster_num)

        indices = [int(id_mapper[str(docid)]) for docid in indice]
        result[qid] = [indices, scores]

        for key, value in time_dict.items():
            local_time_dict[key] += value

    return result, local_time_dict


def multiprocess_search(query_texts, query_embeddings, qid2idx, id_mapper, doer, args):
    num_processes = min(multiprocessing.cpu_count(), 10)  
    chunk_size = len(query_texts) // num_processes  
    chunks = [query_texts[i:i + chunk_size] for i in range(0, len(query_texts), chunk_size)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_queries, 
            [(chunk, query_embeddings, qid2idx, id_mapper, doer, args) for chunk in chunks]
        )
    final_res = {}
    total_time_dict = {
            "total": 0, "sptag": 0, "retrieve_postings1": 0, "merge_postings1": 0,
            "retrieve_postings2": 0, "merge_postings2": 0, "get_topk": 0,
        }

    for partial_res, partial_time_dict in results:
        final_res.update(partial_res)
        for key, value in partial_time_dict.items():
            total_time_dict[key] += value

    return final_res, total_time_dict

def search(query_texts, query_embeddings, qid2idx, id_mapper, doer, args):
    cnt = 0
    # res = defaultdict(dict)
    res = {}
    total_time_dict = {
            "total": 0, "sptag": 0, "retrieve_postings1": 0, "merge_postings1": 0,
            "retrieve_postings2": 0, "merge_postings2": 0, "get_topk": 0,
        }
    for query_text in tqdm(query_texts, total=total_query, desc="Retrieving"):
        qid = int(query_text["text_id"])
        query_embedding = query_embeddings[qid2idx[qid]]
        indice, scores, time_dict = doer.search(query_text, query_embedding, 
                                                    topk=args.depth,
                                                    cluster_num=args.cluster_num) 

        indices = [int(id_mapper[str(docid)]) for docid in indice]
        res[qid] = [indices, scores]

        for key, value in time_dict.items():
            total_time_dict[key] += value
        cnt += 1
        if cnt >= 5:
            break
    return res, total_time_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--build_spann_ii", action="store_true")
    parser.add_argument("--hybrid_search", action="store_true")
    
    parser.add_argument("--cluster_file", type=str, required=False)
    parser.add_argument("--spann_index_dir", type=str, required=False)
    parser.add_argument("--spann_index_name", type=str, required=False, default="invert_index")
    parser.add_argument("--splade_index_dir", type=str, required=False)
    parser.add_argument("--splade_index_name", type=str, required=False, default="invert_index")
    parser.add_argument("--sptags_index_path", type=str, required=False)
    
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--save_method", type=str, default="bin", required=False)

    parser.add_argument("--splade_weight", type=float, default=1, required=False)
    parser.add_argument("--spann_weight", type=float, default=10000, required=False)
    parser.add_argument("--cluster_num", type=int, default=256, required=False)
    parser.add_argument("--depth", type=int, default=1000, required=False)

    parser.add_argument("--query_text_path", type=str, required=False)
    parser.add_argument("--query_emb_path", type=str, required=False)
    parser.add_argument("--doc_emb_path", type=str, required=False)
    parser.add_argument("--qlookup_path", type=str, required=False)
    parser.add_argument("--plookup_path", type=str, required=False)

    parser.add_argument("--use_cmp", action="store_true", required=False)
    parser.add_argument("--use_v2", action="store_true", required=False)
    parser.add_argument("--only_spann_list", action="store_true", required=False)
    parser.add_argument("--v2_dis_method", type=str, required=False, default="rep")
    
    parser.add_argument("--save_rank", action="store_true", required=False)
    parser.add_argument("--save_dict", action="store_true", required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False, default=None)


    args = parser.parse_args()

    if args.build_spann_ii:
        os.makedirs(args.spann_index_dir, exist_ok=True)
        build_invert_index(args)
    
    if args.hybrid_search:
        query_texts, query_embeddings, qlookup = prepare_query(args.query_text_path, args.query_emb_path, args.qlookup_path)
        qid2idx = {}
        for idx, qid in enumerate(qlookup):
            qid2idx[qid] = idx
        total_query = len(query_texts)
        print(query_embeddings.shape)
        id_mapper = json.load(open(args.plookup_path))
        
        doer = Searcher.build(args.splade_index_dir, args.splade_index_name,
                            args.spann_index_dir, args.spann_index_name, 
                            args.sptags_index_path, args.doc_emb_path,
                            args.splade_weight, args.spann_weight,
                            args.use_cmp, args.use_v2, args.v2_dis_method)

        
        
        # res, total_time_dict = search(query_texts, query_embeddings, qid2idx, id_mapper, doer, args)
        res, total_time_dict = multiprocess_search(query_texts, query_embeddings, qid2idx, id_mapper, doer, args)
    
        for key, value in total_time_dict.items():
            print("{}:\t {} ms".format(key, 1000 * value / total_query))      
        
        res_dict = compute_metrics(load_gt(args.gt_path), res)
        print(res_dict)