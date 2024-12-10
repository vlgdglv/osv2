import os
import sys

sys.path += ['../']
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import json
import logging
import argparse
import csv
import numpy as np
import torch
import lmdb
from tqdm import tqdm
import pickle
from transformers import (
    BertTokenizer
)

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


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

def loads_data(buf):
    return pickle.loads(buf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_path", type=str, default=None)
    parser.add_argument("--max_seq_length", default=512, type=int)

    args = parser.parse_args()

    train_doc_pool_txn, test_doc_pool_txn = load_passages_lmdb(args)
    print(test_doc_pool_txn)

    n_passages = loads_data(test_doc_pool_txn.get(b'__len__'))
    print(n_passages)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    id2id = json.load(open(os.path.join(args.passage_path,'id2id_test.json')))

    dataset = TextDatasetLMDBMeta(0, n_passages, test_doc_pool_txn, tokenizer, args, id2id)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=32)
    for idx in tqdm(range(n_passages)):
        cont = dataset[idx]
        