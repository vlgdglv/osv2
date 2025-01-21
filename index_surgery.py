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
from collections import defaultdict
from numba import njit
from numba.typed import Dict, List
from numba import types
import multiprocessing

from inverted_index import InvertedIndex, BM25Retriever





def test_index():
    index_dir = "/datacosmos/User/baoht/onesparse2/marcov2/index/bm25_test"
    index_name = "bm25_inverted_index.bin"
    query_text_path = "/datacosmos/User/baoht/onesparse2/marcov2/index/bm25_test/query_ids.test.json"
    bm25_index = BM25Retriever(index_dir, index_name)
    invert_index = bm25_index.invert_index
    invert_index.engage_numba()
    
    with open(query_text_path, "r") as f:
        query_texts = [json.loads(line.strip()) for line in f]

    for query_line in tqdm(query_texts,desc="Retrieving"):
        qid = int(query_line["text_id"])
        query_text = query_line["text"]
        query_value = [1.0 for _ in range(len(query_text))]
        invert_index.match_and_merge(
            invert_index.numba_index_ids,
            invert_index.numba_index_values,
            101070374,
            np.array(query_text, dtype=np.int32),
            np.array(query_value, dtype=np.float32),
        )

        cnt += 1
        if cnt >= 5:
            break

def modify_index():
    index_dir = "/datacosmos/User/baoht/onesparse2/marcov2/index/bm25_test"
    index_name = "bm25_inverted_index.bin"
    bm25_index = BM25Retriever(index_dir, index_name)
    invert_index = bm25_index.invert_index
    length_dict = {}
    for k, v in invert_index.index_ids.items():
        length_dict[k] = len(v)

    total = invert_index.total_docs
    print(total)
    ratio = 0.1
    thr = total * ratio
    clean_list = []
    for k, v in length_dict.items():
        if v > thr:
            clean_list.append(k)
    print(len(clean_list))
    invert_index.delete_item(clean_list)
    save_dir = "/datacosmos/User/baoht/onesparse2/marcov2/index/bm25_test_cut{}/".format(ratio)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'bm25_inverted_index.bin')
    invert_index.file_path = save_path
    invert_index.save()


def merge_index():
    index_dir = "/datacosmos/User/baoht/onesparse2/marcov2/warehouse/splade_index/cotrain_exp0117"
    shard_count = 5

    full_index = InvertedIndex(index_path="/datacosmos/User/baoht/onesparse2/marcov2/warehouse/splade_index/cotrain_exp0117/",
                               file_name="splade_full.bin",
                               ignore_keys=False,
                               force_rebuild=True)
    
    for shard_id in range(shard_count):
        shard_index = InvertedIndex(index_path=os.path.join(index_dir, f"shard_{shard_id}"), 
                      file_name="splade_index.bin",
                      ignore_keys=False)
        full_index.total_docs += shard_index.total_docs
        key_list = shard_index.index_ids.keys()
        for key in tqdm(key_list):
            shard_ids = shard_index.index_ids[key]
            shard_values = shard_index.index_values[key]
            
            if key in full_index.index_ids.keys():
                full_index.index_ids[key].extend(shard_ids)
                full_index.index_values[key].extend(shard_values)
                
            else:
                full_index.index_ids[key] = list(shard_ids)
                full_index.index_values[key] = list(shard_values)

        print("shard {} done".format(shard_id))
    full_index.save()

def merge_docid():
    index_dir = "/datacosmos/User/baoht/onesparse2/marcov2/warehouse/splade_index/cotrain_exp0117"
    shard_count = 5

    doc_ids_full = []
    for shard_id in range(shard_count):
        doc_ids = pickle.load(open(os.path.join(index_dir, "shard_{}/doc_ids_splade_index.bin.pkl".format(shard_id)), "rb"))
        doc_ids_full.extend(doc_ids)

    save_name = "doc_ids_splade_full.bin.pkl"
    print(len(doc_ids_full))
    pickle.dump(doc_ids_full, open(os.path.join(index_dir, save_name), "wb"))


if __name__ == "__main__":
    # test_index()
    # modify_index()
    merge_index()
    # test_merge()
    # merge_docid()