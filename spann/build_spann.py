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
from ..auxiliary import read_fbin


logger = logging.getLogger(__name__)

 
def build_index(corpus_name, corpus):
    '''
    if corpus_name == "nq":
        with open(corpus_path, "rb") as f:
            corpus, _ = pickle.load(f)
        corpus = np.array(corpus)
    elif corpus_name == "msmarco":
        corpus = []
        for i in range(10):
            with open(corpus_path + "%d.pt" % i, "rb") as f:
                embedding, _ = pickle.load(f)
                corpus.append(embedding)
        corpus = np.vstack(corpus)
    else:
        raise NotImplementedError
    '''

    # corpus = read_fbin(corpus_path)

    vector_number, vector_dim = corpus.shape
    # print(vector_number, vector_dim)

    print("Build index Begin !!!")
    index = SPTAG.AnnIndex('SPANN', 'Float', vector_dim)
    
    index.SetBuildParam("IndexAlgoType", "BKT", "Base")
    index.SetBuildParam("IndexDirectory", corpus_name + "_8", "Base")
    index.SetBuildParam("DistCalcMethod", "L2", "Base")

    index.SetBuildParam("isExecute", "true", "SelectHead")
    index.SetBuildParam("TreeNumber", "1", "SelectHead")
    index.SetBuildParam("BKTKmeansK", "32", "SelectHead")
    index.SetBuildParam("BKTLeafSize", "8", "SelectHead")
    index.SetBuildParam("SamplesNumber", "10000", "SelectHead")
    index.SetBuildParam("SelectThreshold", "50", "SelectHead") 
    index.SetBuildParam("SplitFactor", "6", "SelectHead")    
    index.SetBuildParam("SplitThreshold", "100", "SelectHead")  
    index.SetBuildParam("Ratio", "0.1", "SelectHead")   
    index.SetBuildParam("NumberOfThreads", "16", "SelectHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "SelectHead")

    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
    index.SetBuildParam("TPTNumber", "64", "BuildHead")
    index.SetBuildParam("TPTLeafSize", "2000", "BuildHead")
    index.SetBuildParam("MaxCheck", "8192", "BuildHead")
    index.SetBuildParam("MaxCheckForRefineGraph", "8192", "BuildHead")
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", "16", "BuildHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "BuildHead")

    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", "96", "BuildSSDIndex")
    index.SetBuildParam("NumberOfThreads", "16", "BuildSSDIndex")
    index.SetBuildParam("MaxCheck", "8192", "BuildSSDIndex")

    index.SetBuildParam("SearchPostingPageLimit", "96", "BuildSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("MaxDistRatio", "1000.0", "BuildSSDIndex")

    index.SetBuildParam("MaxCheck", "8192", "SearchSSDIndex")
    index.SetBuildParam("NumberOfThreads", "1", "SearchSSDIndex")
    index.SetBuildParam("SearchPostingPageLimit", "96", "SearchSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "64", "SearchSSDIndex")
    index.SetBuildParam("MaxDistRatio", "1000.0", "SearchSSDIndex")

    if index.Build(corpus, vector_number, False):
        index.Save(corpus_name + "_8")  # Save the index to the disk

    print("Build index accomplished.")


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


def load_passage_embedding_runner(args):
    passage_dir = args.embedding_dir
    for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
        passage_embedding_list = []
        passage_embedding_id_list = []
        for piece in tqdm(range(100)):
            pickle_path = os.path.join(passage_dir,
                                "{1}_data_obj_{0}_piece_{2}.pb".format(str(i), 'passage_embedding',str(piece)))
            pickle_path_id = os.path.join(passage_dir,
                                "{1}_data_obj_{0}_piece_{2}.pb".format(str(i), 'passage_embedding_id',str(piece)))
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
        yield passage_embedding, passage_embedding_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action="store_true")
    parser.add_argument('--search', action="store_true")
    
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--corpus_path', type=str, default=None)
    parser.add_argument("--embedding_dir", type=str, required=False)
    parser.add_argument("--spann_index", type=str, required=False)
    parser.add_argument("--query_emb_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--total_docs", type=int, required=False, default=None)
    parser.add_argument("--depth", type=int, required=False, default=1000)
    args = parser.parse_args()

    passage_embedding = read_fbin(args.corpus_path)

    print("Passage loaded, shape: {}".format(passage_embedding.shape))
    build_index(args.name, passage_embedding)
    