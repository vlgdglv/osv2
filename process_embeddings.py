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

import torch
import struct
from typing import Dict, List, Any, Union, Mapping


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)
 

def load_passage_embedding_runner(embedding_dir, world_size):
    passage_dir = embedding_dir
    for i in range(world_size):  # TODO: dynamically find the max instead of HardCode
        passage_embedding_list = []
        passage_embedding_id_list = []
        for piece in tqdm(range(100)):
            pickle_path = os.path.join(passage_dir,
                                "{1}_test_data_obj_{0}_piece_{2}.pb".format(str(i), 'passage_embedding',str(piece)))
            pickle_path_id = os.path.join(passage_dir,
                                "{1}_test_data_obj_{0}_piece_{2}.pb".format(str(i), 'passage_embedding_id',str(piece)))
            
            if not os.path.isfile(pickle_path):
                break
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_list.append(b)
            with open(pickle_path_id, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_id_list.append(b)
        passage_embedding = np.concatenate(passage_embedding_list, axis=0)
        passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
        print("Chunks loaded: {}".format(passage_embedding.shape))
        yield passage_embedding, passage_embedding_id
        

if __name__ == "__main__":
    embedding_dir = "/home/aiscuser/osv2/embeddings/14000"
    embedding_list, passage_embedding2id_list = [], []
    embedding_and_id_generator = load_passage_embedding_runner(embedding_dir, world_size=7) 
    for k, (passage_embedding, passage_embedding2id) in enumerate(embedding_and_id_generator):
        embedding_list.append(passage_embedding)
        passage_embedding2id_list.append(passage_embedding2id)
    passage_embedding = np.concatenate(embedding_list, axis=0)
    passage_embedding2id_list = np.concatenate(passage_embedding2id_list, axis=0)

    print("Test corpus loaded: ", passage_embedding.shape)
    output_dir = "/home/aiscuser/osv2/embeddings/bs_SimANS_36k"
    write_fbin(os.path.join(output_dir, "test_corpus.bin"), passage_embedding)
    print("corpus saved.")
    with open(os.path.join(output_dir, "plookup.pkl"), 'wb') as f:
        pickle.dump(passage_embedding2id_list, f)
    print("pid saved, shape: {}".format(passage_embedding2id_list.shape))

