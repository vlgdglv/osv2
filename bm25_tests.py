import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import logging

import pytrec_eval
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
logger = logging.getLogger(__name__)

from inverted_index import BM25Retriever
from ms_datasets import TextDatasetLMDBMeta, loads_data, load_passages_lmdb

def get_iterator(args):
    train_doc_pool_txn, test_doc_pool_txn = load_passages_lmdb(args)
    n_passages = loads_data(test_doc_pool_txn.get(b'__len__'))
    print("Total passages: ", n_passages)

    id2id = json.load(open(os.path.join(args.passage_path,'id2id_test.json')))
    dataset = TextDatasetLMDBMeta(0, n_passages, test_doc_pool_txn, tokenizer, args, id2id)

    for idx in tqdm(range(n_passages)):
        cont = dataset[idx]
        yield cont

def eval_with_pytrec(runs, qrel_path, output_path=None):
    with open(qrel_path, 'r') as f:
        qrel_data = f.readlines()

    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        passage = line[1]
        rel = int(1)
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        qrels[query][passage] = rel

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.10", 
                                                       "recall.25", "recall.50",  
                                                       "recall.100", "recall.200", "recall.1000"
                                                       })
    res = evaluator.evaluate(runs)

    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_25_list = [v['recall_25'] for v in res.values()]
    recall_50_list = [v['recall_50'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_200_list = [v['recall_200'] for v in res.values()]
    recall_1000_list = [v['recall_1000'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.10"})
    res = evaluator.evaluate(runs)
    ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@25": np.average(recall_25_list),
            "Recall@50": np.average(recall_50_list),
            "Recall@100": np.average(recall_100_list),
            "Recall@200": np.average(recall_200_list),
            "Recall@1000": np.average(recall_1000_list),
            "NDCG@10": np.average(ndcg_10_list), 
        }
        
    print("---------------------Evaluation results:---------------------")    
    print(res)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(json.dumps(res, indent=4))
        return res

def text_clean(text):
    text = text.replace("#N#", " ")
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, required=False, default="bert-base-multilingual-uncased")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--passage_path", type=str, required=False)
    parser.add_argument("--index_path", type=str, required=False)
    parser.add_argument("--index_name", type=str, required=False)
    parser.add_argument("--do_tokenize", action="store_true")
    parser.add_argument("--index_file_name", type=str, default="array_index", required=False)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--max_seq_length", default=512, type=int)
    
    # BM25 params
    parser.add_argument("--method", type=str, default="lucene", required=False)
    parser.add_argument("--k1", type=float, default=1.2, required=False)
    parser.add_argument("--b", type=float, default=0.75, required=False)
    parser.add_argument("--delta", type=float, default=0.5, required=False)
    
    parser.add_argument("--do_retrieve", action="store_true")
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False)

    args = parser.parse_args()

    print(args)
    if args.build_index:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        bm25 = BM25Retriever(args.index_path, args.index_name, method=args.method, k1=args.k1, b=args.b, delta=args.delta, 
                             force_rebuild=args.force_rebuild)

        corpus_idx, corpus_list = [], []
        if args.do_tokenize:
            runner = get_iterator(args)
            for thing in runner:
                doc_id, body = thing
                ids = tokenizer.encode(body, add_special_tokens=False)
                body = text_clean(body)
                corpus_idx.append(doc_id)
                corpus_list.append(ids)
        else:
            with open(args.passage_path, "r") as f:
                for line in tqdm(f, desc="Loading"):
                    content = json.loads(line.strip())
                    corpus_idx.append(int(content["text_id"]))
                    corpus_list.append(content["text"])
        print("Corpus Loaded: {}".format(len(corpus_idx)))
        # print("Text encode done")
        bm25.index(corpus_idx, corpus_list, tokenizer.vocab.values())

    if args.do_retrieve:
        # stemmer = Stemmer.Stemmer("english")
        # token = Tokenizer(tokenizer_path=args.tokenizer_path, stemmer=stemmer)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        bm25 = BM25Retriever(args.index_path, args.index_name, k1=args.k1, b=args.b, delta=args.delta)

        query_idx, query_list = [], []
        if args.do_tokenize:
            with open(args.query_path, "r", encoding="utf-8") as f:
                for line in f:
                    qid, text = line.strip().split('\t')
                    query_idx.append(qid)
                    query_list.append(tokenizer.encode(text))
        else:
            with open(args.query_path, "r") as f:
                for line in f:
                    content = json.loads(line.strip())
                    query_idx.append(int(content["text_id"]))
                    query_list.append(content["text"])
        print("Query Loaded: {}".format(len(query_idx)))
        
        res = defaultdict(dict)
        bm25.invert_index.engage_numba()
        # with open(args.output_path, "w", encoding="utf-8") as out:
        for qid, query in tqdm(zip(query_idx, query_list), desc="Retrieving", total=len(query_idx)):
            indices, scores = bm25.retrieve(np.array(query))
            for rank, (docid, score) in enumerate(zip(indices, scores)):
                # out.write(f"{qid}\t{docid}\t{rank+1}\t{score}\n")
                res[str(qid)][str(docid)] = float(score)
    
        eval_with_pytrec(res, args.gt_path)