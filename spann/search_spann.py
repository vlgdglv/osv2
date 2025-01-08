import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import csv
import json
import argparse
import logging
import pickle
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import argparse
import SPTAG

from collections import defaultdict
from auxiliary import read_fbin_mmap, read_fbin, read_ibin, compute_metrics, load_gt


def sptag_search(index, embedding, depth):
    """
    return a tuple(cluster_id, score), latency
    """
    start = timer()
    # top 256 clusters
    result = index.Search(embedding, depth) # the search results are not docid, but cluster id
    latency = timer() - start
    return [(result[0][i], 1.0/(1.0+float(result[1][i]))) for i in range(len(result[1]))], latency


def prepare_query(query_text_path, query_emb_path):
    with open(query_text_path, "r") as f:
        query_texts = [json.loads(line.strip()) for line in f]
    with open(query_emb_path, "rb") as f:
        query_embeddings, qid = pickle.load(f)
    return query_texts, query_embeddings, qid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--spann_index", type=str, required=False)
    parser.add_argument("--qlookup_path", type=str, required=False)
    parser.add_argument("--plookup_path", type=str, required=False)
    parser.add_argument("--query_emb_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--total_docs", type=int, required=False, default=None)
    parser.add_argument("--depth", type=int, required=False, default=100)
    args = parser.parse_args()

    sptag_index = SPTAG.AnnIndex.Load(args.spann_index)
    
    query_embeddings = read_fbin(args.query_emb_path)
    qlookup = read_ibin(args.qlookup_path) 
    
    total_query = len(query_embeddings)
    
    res = {}
    plookup_type = args.plookup_path.split(".")[-1]
    if plookup_type == "json":
        id_mapper = json.load(open(args.plookup_path))
    # with open(args.output_path, "w") as f:
    for query_embedding, qid in tqdm(zip(query_embeddings, qlookup), total=total_query, desc="Retrieving"):
        cluster_list, latency = sptag_search(sptag_index, query_embedding, args.depth)
        cid_list, cid_score = [cluster[0] for cluster in cluster_list], [cluster[1] for cluster in cluster_list]
        indices = [int(id_mapper[str(docid)]) for docid in cid_list]
        if isinstance(qid, list):
            qid = qid[0]
        res[int(qid[0])] = [indices, cid_score]

    res_dict = compute_metrics(load_gt(args.gt_path), res)
    print(res_dict)