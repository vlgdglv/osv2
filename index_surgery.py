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

if __name__ == "__main__":
    # test_index()
    modify_index()