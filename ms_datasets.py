import os
import sys

sys.path += ['../']
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import json
import logging
import argparse
import struct
import csv
import numpy as np
import torch
import lmdb
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from transformers import (
    BertTokenizer
)

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def text_clean(text):
    return " ".join(
        text.replace("#n#", " ")
            .replace("#N#", " ")
            .replace("<sep>", " ")
            .replace("#tab#", " ")
            .replace("#r#", " ")
            .replace("\t", " ")
            .split()
    )


class TextDatasetLMDBMeta(torch.utils.data.Dataset):
    def __init__(self, start_idx, end_idx, doc_pool_txn, tokenizer, opt, id2id=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.doc_pool_txn = doc_pool_txn
        self.length = self.end_idx - self.start_idx
        self.tokenizer = tokenizer
        self.opt = opt
        self.max_seq_length = opt.max_seq_length
        self.id2id = id2id

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.id2id!=None:
            real_index = str(self.id2id[str(index + self.start_idx)])
        else:
            real_index = str(index + 1 + self.start_idx)
        url, title, body = pickle.loads(self.doc_pool_txn.get(real_index.encode()))
        
        title, body = text_clean(title), text_clean(body)
        prev_tokens = ['[CLS]']  + self.tokenizer.tokenize(url)[:42] + ['[SEP]'] + self.tokenizer.tokenize(title)[:41] + ['[SEP]']
        body_tokens = self.tokenizer.tokenize(body)[:(self.max_seq_length - len(prev_tokens) - 1)]
        passage = prev_tokens + body_tokens + ['[SEP]']
        passage = self.tokenizer.convert_tokens_to_ids(passage)[:self.max_seq_length]

        return real_index, passage

    @classmethod
    def get_collate_fn(cls, args):
        def create_passage_input(features):
            index_list = [x[0] for x in features]
            d_list = [x[1] for x in features]
            max_d_len = max([len(d) for d in d_list])
            d_list = [d + [args.pad_token_id] * (max_d_len - len(d)) for d in d_list]
            doc_tensor = torch.LongTensor(d_list)
            return index_list, doc_tensor, (doc_tensor != 0).long()
        return create_passage_input


def load_passages_lmdb(args):
    if not os.path.exists(args.passage_path):
        logger.info(f'{args.passage_path} does not exist')
        return
    passages_train_path = os.path.join(args.passage_path, 'test_lmdb_new')
    passages_test_path = os.path.join(args.passage_path, 'test_lmdb_new')

    logger.info(f'Loading passages from: {passages_train_path}')
    doc_pool_env = lmdb.open(passages_train_path, subdir=os.path.isdir(passages_train_path), readonly=True, lock=False, readahead=False, meminit=False)
    logger.info(f'Loading passages from: {passages_test_path}')
    doc_pool_env_test = lmdb.open(passages_test_path, subdir=os.path.isdir(passages_test_path), readonly=True, lock=False, readahead=False, meminit=False)
    return doc_pool_env.begin(write=False), doc_pool_env_test.begin(write=False)


class TextDatasetLMDBMeta(torch.utils.data.Dataset):
    def __init__(self, start_idx, end_idx, doc_pool_txn, tokenizer, opt, id2id=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.doc_pool_txn = doc_pool_txn
        self.length = self.end_idx - self.start_idx
        self.tokenizer = tokenizer
        self.opt = opt
        self.max_seq_length = opt.max_seq_length
        self.id2id = id2id

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.id2id!=None:
            real_index = str(self.id2id[str(index + self.start_idx)])
        else:
            real_index = str(index + 1 + self.start_idx)
        url, title, body = pickle.loads(self.doc_pool_txn.get(real_index.encode()))

        prev_tokens = ['[CLS]']  + self.tokenizer.tokenize(url)[:42] + ['[SEP]'] + self.tokenizer.tokenize(title)[:41] + ['[SEP]']
        body_tokens = self.tokenizer.tokenize(body)[:(self.max_seq_length - len(prev_tokens) - 1)]
        passage = prev_tokens + body_tokens + ['[SEP]']

        passage = self.tokenizer.convert_tokens_to_ids(passage)[:self.max_seq_length]
        
        return real_index, passage

    @classmethod
    def get_collate_fn(cls, args):
        def create_passage_input(features):
            index_list = [x[0] for x in features]
            d_list = [x[1] for x in features]
            max_d_len = max([len(d) for d in d_list])
            d_list = [d + [args.pad_token_id] * (max_d_len - len(d)) for d in d_list]
            doc_tensor = torch.LongTensor(d_list)
            return index_list, doc_tensor, (doc_tensor != 0).long()
        return create_passage_input

class TextIdsDatasetLMDBMeta(torch.utils.data.Dataset):
    def __init__(self, start_idx, end_idx, doc_pool_txn, opt, id2id=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.doc_pool_txn = doc_pool_txn
        self.length = self.end_idx - self.start_idx
        self.opt = opt
        self.max_seq_length = opt.max_seq_length
        self.id2id = id2id

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.id2id!=None:
            real_index = str(self.id2id[str(index + self.start_idx)])
        else:
            real_index = str(index + 1 + self.start_idx)
        text = pickle.loads(self.doc_pool_txn.get(real_index.encode()))
        
        return real_index, text

    @classmethod
    def get_collate_fn(cls, args):
        def create_passage_input(features):
            index_list = [x[0] for x in features]
            d_list = [x[1] for x in features]
            max_d_len = max([len(d) for d in d_list])
            d_list = [d + [args.pad_token_id] * (max_d_len - len(d)) for d in d_list]
            doc_tensor = torch.LongTensor(d_list)
            return index_list, doc_tensor, (doc_tensor != 0).long()
        return create_passage_input



def loads_data(buf):
    return pickle.loads(buf)

def tokenize_query(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    assert args.save_path is not None
    if False:
        with open(args.query_path, "r", encoding="utf-8") as fi, open(args.save_path, "w") as fo:
            for line in fi:
                qid, text, lang = line.strip().split('\t')
                ids = tokenizer.encode(text)
                fo.write(json.dumps({"qid": qid, "text_ids": ids}) + "\n")
    if True:
        with open(args.query_path, "r", encoding="utf-8") as fi, open(args.save_path, "wb") as fo:
            lines = fi.readlines()
            fo.write(struct.pack("I", len(lines)))
            for line in lines:
                qid, text, lang = line.strip().split('\t')
                ids = np.array(list(tokenizer.encode(text)), np.int32)
                fo.write(struct.pack("I I I", int(qid), len(ids), ids.nbytes))
                fo.write(ids.tobytes())

def pack_gts(args):
    gt_dict = {}
    with open(args.gt_path, 'r') as fi:
        for l in fi:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                if qid in gt_dict:
                    pass
                else:
                    gt_dict[qid] = []
                gt_dict[qid].append(int(l[2]))
            except:
                raise IOError('\"%s\" is not valid format' % l)
    with open(args.save_path, "wb") as fo:
        fo.write(struct.pack("I", len(gt_dict)))
        for k, v in gt_dict.items():
            arr_v = np.array(v, np.int32)
            fo.write(struct.pack("I I I", int(k), len(arr_v), arr_v.nbytes))
            fo.write(arr_v.tobytes())


def sample_corpus(args):
    idmapper = json.load(open(args.id_mapper_path, 'r'))
    print("idmapper length:", len(idmapper))
    sample_num = int(args.total_docs * args.sample_ratio)

    gt_docis = []
    with open(args.gt_path, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            gt_docis.append(int(l[2]))

    sample_ids = np.random.choice(range(len(idmapper)), sample_num, replace=False)
    
    sample_docid = set()
    for i in tqdm(sample_ids):
        sample_docid.add(int(idmapper[str(i)]))

    for docid in gt_docis:
        if docid not in sample_docid:
            sample_docid.add(docid)
    
    sample_docid = list(sample_docid)
    print("sample docid length:", len(sample_docid))
    with open(args.save_path, 'wb') as f:
        pickle.dump(sample_docid, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=False, default="bert-base-multilingual-uncased")
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False)
    parser.add_argument("--save_path", type=str, required=False)
    parser.add_argument("--passage_path", type=str, default=None)
    parser.add_argument("--id_mapper_path", type=str, default=None)
    parser.add_argument("--total_docs", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--max_seq_length", default=128, type=int)

    args = parser.parse_args()
    # pack_gts(args)
    sample_corpus(args)
    
