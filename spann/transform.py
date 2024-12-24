# Since inner product does not support triangular inequality, 
# we need to transform dense embeddings from inner product space to L2 norm space.
# Based on: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-do-max-inner-product-search-on-indexes-that-support-only-l2
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import csv
import numpy as np
import argparse
from collections import OrderedDict
from ..auxiliary import read_fbin, write_fbin

def transform_corpus(corpus_path, output_path):
    # load data
    # with open(corpus_path, 'rb') as f:  
    #     doc, id_list = pickle.load(f)
    
    doc = read_fbin(corpus_path)
    x = doc
    print(x.shape)  # (8841823, 768)

    # x:      np.array, shape=(N, 768)
    # output: np.array, shape=(N, 769)
    norms = np.linalg.norm(x, axis=1)**2
    phi = norms.max()
    extracol = np.sqrt(phi - norms)
    doc_embeddings = np.hstack((extracol.reshape(-1, 1), x)).astype(np.float32)
    print(doc_embeddings.shape) # (8841823, 769)

    # save
    write_fbin(output_path, doc_embeddings)
    # with open(output_path, 'wb') as f:
    #     pickle.dump((doc_embeddings, id_list), f)
    
# for query embeddings
def transform_query(query_path, output_path): 
    # with open(query_path, "rb") as f:
    #     queries, id_list = pickle.load(f)

    queries = read_fbin(query_path)
    x = queries
    print(x.shape)  # (6980, 768)
    # print(x[0][:10])
    # x:      np.array, shape=(N, 768)
    # output: np.array, shape=(N, 769)
    extracol = np.zeros(x.shape[0]).astype(np.float32)
    query_embeddings = np.hstack((extracol.reshape(-1, 1), x)).astype(np.float32)
    print(query_embeddings.shape)   # (6980, 769)
    # print(query_embeddings[0][:10])
    # save
    write_fbin(output_path, query_embeddings)
    # with open(output_path, 'wb') as f:
    #     pickle.dump((query_embeddings, id_list), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", type=str, required=False, default=None)
    parser.add_argument("--corpus_output_path", type=str, required=False, default=None)
    parser.add_argument("--query_path", type=str, required=False, default=None)
    parser.add_argument("--query_output_path", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.corpus_path is not None and args.corpus_output_path is not None:
        print("transforming corpus")
        transform_corpus(args.corpus_path, args.corpus_output_path)
    if args.query_path is not None and args.query_output_path is not None:
        print("transforming querys")
        transform_query(args.query_path, args.query_output_path)
    