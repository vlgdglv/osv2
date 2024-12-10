import os
import h5py
import json
import math
import array
import numba
import struct
import pickle
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from numba.typed import Dict, List
from numba import types

logger = logging.getLogger(__name__)

class InvertedIndex:
    def __init__(self, 
                 index_path=None, 
                 file_name="array_index.h5py",
                 force_rebuild=False,
                 ignore_keys=False, 
                 ignore_token_before=173):
        os.makedirs(index_path, exist_ok=True)
        
        self.file_path = os.path.join(index_path, file_name)
        self.index_path = index_path
        suffix = file_name.split(".")[-1]

        if suffix == "h5py":
            self.save_method = "h5py"
        elif suffix == "pkl":
            self.save_method = "pkl"
        elif suffix == "bin":
            self.save_method = "bin"    
        else:
            raise ValueError("Unsupported save method")
        
        if os.path.exists(self.file_path) and not force_rebuild:
            self.load()
        else:
            self.index_ids = defaultdict(lambda: array.array('i'))
            self.index_values = defaultdict(lambda: array.array('f'))
            self.total_docs = 0        

        self.numba = False
        if ignore_keys:
            self.ignore_keys = ignore_keys
            self.ignore_token_before = ignore_token_before
            
    def load(self, path=None, method=None):
        path = path if path is not None else self.file_path
        method = method if method is not None else self.save_method
        print("Loading index from {}".format(path))
        if method == "h5py":
            with h5py.File(path, "r") as f:
                self.index_ids = dict()
                self.index_values = dict()
                dim = f["dim"][()]
                self.total_docs = f["total_docs"][()]
                for key in tqdm(range(dim), desc="Loading index"):
                    try:
                        self.index_ids[key] = np.array(f["index_ids_{}".format(key)], dtype=np.int32)
                        self.index_values[key] = np.array(f["index_values_{}".format(key)], dtype=np.float32)
                    except:
                        self.index_ids[key] = np.array([], dtype=np.int32)
                        self.index_values[key] = np.array([], dtype=np.float32)
                f.close()
        elif method == "pkl":
            with open(path, "rb") as f:
                index_ids, index_values, total_docs = pickle.load(f)
                self.index_ids, self.index_values, self.total_docs = index_ids, index_values, total_docs
                f.close()
        elif method == "bin":
            with open(path, "rb") as f:
                num_keys, total_docs = struct.unpack('I I', f.read(8))
                index_ids = dict()
                index_values = dict()
                for _ in range(num_keys):
                    key = struct.unpack('I', f.read(4))[0]

                    ids_size = struct.unpack('I', f.read(4))[0]
                    ids_data = f.read(ids_size)
                    ids_array = np.frombuffer(ids_data, dtype=np.int32)

                    values_size = struct.unpack('I', f.read(4))[0]
                    values_data = f.read(values_size)
                    values_array = np.frombuffer(values_data, dtype=np.float32)

                    index_ids[key] = ids_array
                    index_values[key] = values_array

                self.total_docs = total_docs
                self.index_ids, self.index_values = index_ids, index_values
                f.close()
        else:
            raise ValueError("Unsupported save method")
        print("Index loaded, total docs: {}".format(self.total_docs))

    def save(self):
        print("Converting to numpy")
        for key in tqdm(list(self.index_ids.keys()), desc="Converting to numpy"):
            sorted_indices = np.argsort(self.index_ids[key])
            self.index_ids[key] = np.array([self.index_ids[key][i] for i in sorted_indices], dtype=np.int32)
            self.index_values[key] = np.array([self.index_values[key][i] for i in sorted_indices], dtype=np.float32)
        
        print("Save index to {}".format(self.file_path))
        if self.save_method == "h5py":
            with h5py.File(self.file_path, "w") as f:
                f.create_dataset("dim", data=len(self.index_ids.keys()))
                f.create_dataset("total_docs", data=self.total_docs)    
                for key in tqdm(self.index_ids.keys(), desc="Saving"):
                    f.create_dataset("index_ids_{}".format(key), data=self.index_ids[key])
                    f.create_dataset("index_values_{}".format(key), data=self.index_values[key])
                f.close()
        elif self.save_method == "pkl":
            with open(self.file_path, "wb") as f:
                pickle.dump((self.index_ids, self.index_values, self.total_docs), f)
        elif self.save_method == "bin":
            with open(self.file_path, "wb") as f:
                dim = len(self.index_ids.keys())
                f.write(struct.pack("I I", dim, self.total_docs))
                for key in tqdm(self.index_ids.keys(), desc="Saving"):
                    ids_array, values_array  = self.index_ids[key], self.index_values[key]
                    ids_size, values_size = ids_array.nbytes, values_array.nbytes

                    f.write(struct.pack("I", key))

                    f.write(struct.pack("I", ids_size))
                    f.write(ids_array.tobytes())

                    f.write(struct.pack("I", values_size))
                    f.write(values_array.tobytes())
        else:
            raise ValueError("Unsupported save method: {}".format(self.save_method))
        
        print("Index saved.")
        index_dist = {}
        for k, v in self.index_ids.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))
        print("Dist dumped.")

    def add_item(self, col, row, value):
        if self.ignore_keys and row < self.ignore_token_before:
            return
        self.index_ids[col].append(int(row))
        self.index_values[col].append(value)
        self.total_docs += 1

    def add_batch_item(self, col, row, value):
        for r, c, v in zip(row, col, value):
            if self.ignore_keys and r < self.ignore_token_before:
                continue
            self.index_ids[c].append(int(r))
            self.index_values[c].append(v)
        self.total_docs += len(set(row))

    def engage_numba(self):
        self.numba_index_ids = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
        self.numba_index_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:]
        )
        self.numba = True

        for k, v in self.index_ids.items():
            self.numba_index_ids[k] = np.array(v, dtype=np.int64)
        for k, v in self.index_values.items():
            self.numba_index_values[k] = np.array(v, dtype=np.float64)
        print("Numba engaged")

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_match(numba_index_ids: numba.typed.Dict,
                    numba_index_values: numba.typed.Dict, 
                    query_ids: np.ndarray, 
                    corpus_size: int, 
                    query_values: np.ndarray = None,
            ):
        scores = np.zeros(corpus_size, dtype=np.float32)
        N = len(query_ids)
        for i in range(N):
            query_idx, query_value = query_ids[i], query_values[i] if query_values is not None else 1.0
            try:
                retrieved_indices = numba_index_ids[query_idx]
                retrieved_values = numba_index_values[query_idx]
            except:
                continue
            for j in numba.prange(len(retrieved_indices)):
                scores[retrieved_indices[j]] += query_value * retrieved_values[j]
        return scores
    
    def select_topk(self, scores, threshold=0.0, topk=100):
        filtered_indices = np.argwhere(scores > threshold)[:, 0]
        scores = scores[filtered_indices]
        if len(scores) > topk:
            top_indices = np.argpartition(scores, -topk)[-topk:]
            filtered_indices, scores = filtered_indices[top_indices], scores[top_indices]
        sorted_indices = np.argsort(-scores)
        return filtered_indices[sorted_indices], scores[sorted_indices]

class BM25Retriever:
    def __init__(self,
                 index_path,
                 index_name,
                 method="lucene",
                 k1=1.2,
                 b=0.75,
                 delta=0.5,
                 force_rebuild=False):
        self.method = method
        self.k1 = k1
        self.b = b
        self.delta = delta

        self.index_path = index_path
        self.invert_index = InvertedIndex(index_path, index_name, force_rebuild=force_rebuild)
        print("BM25 index total docs: {}".format(self.invert_index.total_docs)) 

    def index(self,
              corpus_index,
              corpus_ids,
              vocab_ids):
        n_docs, n_vocab = len(corpus_ids), len(vocab_ids)
        print("Total docs: {}, vocab size: {}".format(n_docs, n_vocab))
        avg_doc_len = np.mean([len(doc) for doc in corpus_ids])

        doc_frequencies = self._calc_doc_frequencies(corpus_ids, vocab_ids)
        idf_array = self._calc_idf_array(select_idf_scorer(self.method), doc_frequencies, n_docs)

        calc_tfc_fn = select_tfc_scorer(self.method)

        for doc_idx, token_ids in tqdm(zip(corpus_index, corpus_ids), desc="BM25 Indexing", total=n_docs):
            doc_len = len(token_ids)

            unique_tokens = set(token_ids)
            tf_dict = {tid: token_ids.count(tid) for tid in unique_tokens}
            token_in_doc = np.array(list(tf_dict.keys()))
            if len(token_in_doc) == 0:
                self.invert_index.add_batch_item([0], [0], [0])
                continue
            tf_array = np.array(list(tf_dict.values()))

            tfc = calc_tfc_fn(tf_array, doc_len, avg_doc_len, self.k1, self.b, self.delta)
            idf = idf_array[token_in_doc]

            scores = tfc * idf 
            self.invert_index.add_batch_item(token_in_doc, [doc_idx for _ in range(len(token_in_doc))], scores)

        self.invert_index.save()
    
    def _calc_doc_frequencies(self, corpus_ids, vocab_ids):
        vocab_set = set(vocab_ids)
    
        doc_freq_dict = {token_id: 0 for token_id in vocab_ids}

        for doc in corpus_ids:
            for doc_token in vocab_set.intersection(set(doc)):
                doc_freq_dict[doc_token] += 1
        return doc_freq_dict
    
    def _calc_idf_array(self, idf_calc_fn, doc_frequencies, n_docs):
        idf_array = np.zeros(len(doc_frequencies))
        for token_id, doc_freq in doc_frequencies.items():
            idf_array[token_id] = idf_calc_fn(doc_freq, n_docs)

        return idf_array


def _score_idf_robertson(df, N, allow_negative=False):
    """
    Computes the inverse document frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    inner = (N - df + 0.5) / (df + 0.5)
    if not allow_negative and inner < 1:
        inner = 1

    return math.log(inner)


def _score_idf_lucene(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def _score_idf_atire(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(N / df)


def _score_idf_bm25l(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / (df + 0.5))


def _score_idf_bm25plus(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / df)

def _score_tfc_robertson(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return tf_array / (k1 * ((1 - b) + b * l_d / l_avg) + tf_array)


def _score_tfc_lucene(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return _score_tfc_robertson(tf_array, l_d, l_avg, k1, b)


def _score_tfc_atire(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return (tf_array * (k1 + 1)) / (tf_array + k1 * (1 - b + b * l_d / l_avg))


def _score_tfc_bm25l(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    c_array = tf_array / (1 - b + b * l_d / l_avg)
    return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)


def _score_tfc_bm25plus(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    num = (k1 + 1) * tf_array
    den = k1 * (1 - b + b * l_d / l_avg) + tf_array
    return (num / den) + delta

def select_idf_scorer(method) -> callable:
    if method == "robertson":
        return _score_idf_robertson
    elif method == "lucene":
        return _score_idf_lucene
    elif method == "atire":
        return _score_idf_atire
    elif method == "bm25l":
        return _score_idf_bm25l
    elif method == "bm25+":
        return _score_idf_bm25plus
    else:
        error_msg = f"Invalid score_idf_inner value: {method}. Choose from 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'."
        raise ValueError(error_msg)
    
def select_tfc_scorer(method) -> callable:
    if method == "robertson":
        return _score_tfc_robertson
    elif method == "lucene":
        return _score_tfc_lucene
    elif method == "atire":
        return _score_tfc_atire
    elif method == "bm25l":
        return _score_tfc_bm25l
    elif method == "bm25+":
        return _score_tfc_bm25plus
    else:
        error_msg = f"Invalid score_tfc value: {method}. Choose from 'robertson', 'lucene', 'atire'."
        raise ValueError(error_msg)
