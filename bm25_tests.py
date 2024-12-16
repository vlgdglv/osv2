import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import logging

import lmdb
import pytrec_eval
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
logger = logging.getLogger(__name__)

from inverted_index import BM25Retriever
from ms_datasets import TextDatasetLMDBMeta, loads_data, load_passages_lmdb, TextIdsDatasetLMDBMeta

def get_iterator(args):
    passages_path = args.passage_path
    logger.info(f'Loading passages from: {passages_path}')
    doc_pool_env_test = lmdb.open(passages_path, subdir=os.path.isdir(passages_path), readonly=True, lock=False, readahead=False, meminit=False)
    test_doc_pool_txn = doc_pool_env_test.begin(write=False)
    n_passages = loads_data(test_doc_pool_txn.get(b'__len__'))
    print("Total passages: ", n_passages)

    if args.idmap_path:
        id2id = json.load(open(args.idmap_path))
    else:
        id2id = None
    dataset = TextIdsDatasetLMDBMeta(0, n_passages, test_doc_pool_txn, args, id2id)

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
        passage = line[2]
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

def load_reference_from_stream(path_to_reference):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    with open(path_to_reference, 'r') as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                if qid in qids_to_relevant_passageids:
                    pass
                else:
                    qids_to_relevant_passageids[qid] = []
                qids_to_relevant_passageids[qid].append(int(l[2]))
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    MaxMRRRank = 10
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    recall_q_top1 = set()
    recall_q_top5 = set()
    recall_q_top10 = set()
    recall_q_top20 = set()
    recall_q_all = set()

    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1.0 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    recall_q_all.add(qid)
                    if i < 5:
                        recall_q_top5.add(qid)
                    if i < 10:
                        recall_q_top10.add(qid)
                    if i < 20:
                        recall_q_top20.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR / len(qids_to_relevant_passageids)
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_relevant_passageids)
    recall_top5 = len(recall_q_top5) * 1.0 / len(qids_to_relevant_passageids)
    recall_top10 = len(recall_q_top10) * 1.0 / len(qids_to_relevant_passageids)
    recall_top20 = len(recall_q_top20) * 1.0 / len(qids_to_relevant_passageids)

    recall_all = len(recall_q_all) * 1.0 / len(qids_to_relevant_passageids)
    all_scores['MRR @10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@5"] = recall_top5
    all_scores["recall@10"] = recall_top10
    all_scores["recall@20"] = recall_top20
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores

def text_clean(text):
    text = text.replace("#N#", " ")
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, required=False, default="bert-base-multilingual-uncased")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--passage_path", type=str, required=False)
    parser.add_argument("--index_path", type=str, required=False)
    parser.add_argument("--idmap_path", type=str, default=None, required=False)
    parser.add_argument("--index_name", type=str, required=False)
    parser.add_argument("--do_tokenize", action="store_true")
    parser.add_argument("--index_file_name", type=str, default="array_index", required=False)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--max_seq_length", default=128, type=int)
    
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
        runner = get_iterator(args)
        for cont in runner:
            doc_id, text_ids = cont
            corpus_idx.append(doc_id)
            corpus_list.append(text_ids)
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
                    qid, text, lang = line.strip().split('\t')
                    query_idx.append(qid)
                    query_list.append(tokenizer.encode(text))
        else:
            with open(args.query_path, "r") as f:
                for line in f:
                    content = json.loads(line.strip())
                    query_idx.append(int(content["text_id"]))
                    query_list.append(content["text"])
        print("Query Loaded: {}".format(len(query_idx)))
        
        if args.idmap_path is not None:
            idmap = json.load(open(args.idmap_path))
            print("Idmap loaded: {}".format(len(idmap)))
        else:
            idmap = None 
        mapping_func = lambda x: idmap[x] if idmap is not None else x
        res = defaultdict(dict)
        qids_to_ranked_candidate_passages = {}
        bm25.invert_index.engage_numba()
        # with open(args.output_path, "w", encoding="utf-8") as out:
        cnt = 0
        for qid, query in tqdm(zip(query_idx, query_list), desc="Retrieving", total=len(query_idx)):
            indices, scores = bm25.retrieve(np.array(query))
            # for pytrec eval
            # for rank, (docid, score) in enumerate(zip(indices, scores)):
            #     # out.write(f"{qid}\t{docid}\t{rank+1}\t{score}\n")
            #     res[str(qid)][str(mapping_func(str(docid)))] = float(score)
            # for simple eval
            qids_to_ranked_candidate_passages[int(qid)] = [int(mapping_func(str(docid))) for docid in indices]
            # for rank, (docid, score) in enumerate(zip(indices, scores)):
                # out.write(f"{qid}\t{docid}\t{rank+1}\t{score}\n")
            break

            
    # eval_with_pytrec(res, args.gt_path)
    reldict = load_reference_from_stream(args.gt_path)   
    all_scores = compute_metrics(reldict, qids_to_ranked_candidate_passages)
    print(all_scores)
    # print(res)