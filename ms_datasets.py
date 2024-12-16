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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=False, default="bert-base-multilingual-uncased")
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False)
    parser.add_argument("--save_path", type=str, required=False)
    parser.add_argument("--passage_path", type=str, default=None)
    parser.add_argument("--max_seq_length", default=128, type=int)

    args = parser.parse_args()
    # tokenize_query(args)
    pack_gts(args)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    # tokenizer.save_pretrained('bm25_tokenizer')

    # train_doc_pool_txn, test_doc_pool_txn = load_passages_lmdb(args)
    # n_passages = loads_data(test_doc_pool_txn.get(b'__len__'))
    # print("Test collection", n_passages)

    # id2id = json.load(open(os.path.join(args.passage_path,'id2id_test.json')))

    # passages_train_path = "/datacosmos/local/User/baoht/onesparse2/marcov2/data/lmdb_data/test_ids_lmdb_new"
    # # passages_train_path = "/datacosmos/local/User/baoht/onesparse2/marcov2/data/lmdb_data/train_ids_lmdb"
    # logger.info(f'Loading passages from: {passages_train_path}')
    # doc_pool_env = lmdb.open(passages_train_path, subdir=os.path.isdir(passages_train_path), readonly=True, lock=False, readahead=False, meminit=False)
    # txn = doc_pool_env.begin(write=False)
    # stats = doc_pool_env.stat()
    # print("Number of entries:", stats['entries'])
    # n_passages = loads_data(txn.get(b'__len__'))
    # print("Collection size", n_passages)


    # dataset = TextIdsDatasetLMDBMeta(0, 10000000, txn, args)
    # missings_list = []
    # for idx in tqdm(range(n_passages)):
    #     realid, text = dataset[idx]
    #     if text == -1:
    #         missings_list.append(realid)
    # print(missings_list)
    # print("Missings", len(missings_list))
    # print(dataset[0])
    # dataset = TextDatasetLMDBMeta(0, n_passages, test_doc_pool_txn, tokenizer, args, id2id)
    # print(dataset[0])

    # n_passages = loads_data(train_doc_pool_txn.get(b'__len__'))
    # print("Train collection", n_passages)
    # dataset = TextDatasetLMDBMeta(0, n_passages, train_doc_pool_txn, tokenizer, args)
    # print(dataset[0])
    # dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=32)

    # for idx in tqdm(range(n_passages)):
    #     cont = dataset[idx]

        