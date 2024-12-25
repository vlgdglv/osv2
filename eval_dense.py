import os
import sys
import lmdb
import torch
import torch.nn as nn
import logging
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    AutoTokenizer, PreTrainedTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding
)

from model_zoo import BiEncoder
from tqdm import tqdm
from inverted_index import InvertedIndex
from auxiliary import (
    to_device, load_gt, compute_metrics,
    DataConfig, 
    ModelConfig, 
    EvaluationConfig, 
    write_fbin, write_ibin
)

logger = logging.getLogger(__name__)

class MARCOWSTestIdsDataset(Dataset):
    def __init__(self, passage_lmdb_env, tokenizer, 
                 start_idx=0, end_idx=-1,
                 idmapping=None, max_length=128):
        self.passage_lmdb_env = passage_lmdb_env 
        self.start = start_idx
        self.end = end_idx
        self.length = self.end - self.start
        if idmapping:
            self.idmapping = idmapping
            self.mapper = lambda x: self.idmapping[str(x)]
        else:
            self.mapper = lambda x: x+1 # For train
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        real_index = str(self.mapper(self.start + index))
        with self.passage_lmdb_env.begin(write=False) as doc_pool_txn:
            text_ids = pickle.loads(doc_pool_txn.get(real_index.encode()))
        encoded_text = self.tokenizer.prepare_for_model(
            text_ids,
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            add_special_tokens=True
        )
        return real_index, encoded_text

@dataclass
class PredictionCollator(DataCollatorWithPadding):
    is_query: bool = False
    q_max_len: int = 32
    k_max_len: int = 128
    def __call__(self, features):
        text_ids, encode_texts = [f[0] for f in features], [f[1] for f in features]
        collated_texts = super().__call__(encode_texts)
        return text_ids, collated_texts


def inference(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()
    if device == torch.device("cuda") and gpu_num > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    model.to(device)
    model.eval()

    id_list, embeddings_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_id, encode_passages = batch
            encode_passages = to_device(encode_passages, device)
            outputs = model(**encode_passages)
            
            # sent_emb = outputs.last_hidden_state[:, 0]
            sent_emb = outputs["sent_emb"]
            sent_emb = sent_emb.cpu().numpy()
            id_list.extend(text_id)
            embeddings_list.extend(sent_emb)
    
    id_list, embeddings_list = np.array(id_list, dtype=np.int32), np.array(embeddings_list, dtype=np.float32)
    id_list = np.reshape(id_list, (-1, 1))
    return id_list, embeddings_list

def store_embeddings(id_list, embeddings_list, output_path, name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_fbin(os.path.join(output_path, "{}_embeddings.fbin".format(name)), embeddings_list)
    write_ibin(os.path.join(output_path, "{}_ids.ibin".format(name)), id_list)

def eval_dense():
    parser = parser = HfArgumentParser((EvaluationConfig, DataConfig, ModelConfig))

    eval_config, data_config, model_config = parser.parse_args_into_dataclasses()
    eval_config: EvaluationConfig
    data_config: DataConfig
    model_config: ModelConfig
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.info("MODEL parameters %s", model_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name,
        cache_dir=model_config.model_cache_dir 
    )
    model_config.vocab_size = len(tokenizer)
    eval_config.vocab_size = len(tokenizer)
    model_config.half = eval_config.fp16


    full_model = BiEncoder(model_config)
    logger.info("Model loaded.")
    
    if eval_config.encode_query:
        logger.info("--------------------- ENCODE QUERY PROCEDURE ---------------------")
        is_query = True
        query_lmdb_env = lmdb.open(data_config.query_lmdb_dir, subdir=os.path.isdir(data_config.query_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        with query_lmdb_env.begin(write=False) as txn:
            n_query = pickle.loads(txn.get(b'__len__'))

        query_dataset = MARCOWSTestIdsDataset(query_lmdb_env, tokenizer,
                                               start_idx=0, end_idx=n_query,
                                               max_length=data_config.q_max_len)
        
        query_loader = DataLoader(
            query_dataset,
            batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
            collate_fn=PredictionCollator(
                tokenizer=tokenizer,
                max_length=data_config.q_max_len,
                is_query=is_query
            ),
            num_workers=eval_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        model = full_model.q_encoder

        id_list, embeddings_list = inference(model, query_loader)
        logger.info(f"Query embeddings shape: {embeddings_list.shape}")
        
        store_embeddings(id_list, embeddings_list, eval_config.embedding_output_dir, "query")
        
    if eval_config.encode_corpus:
        logger.info("--------------------- ENCODE CORPUS PROCEDURE ---------------------")
        is_query = False
        model = full_model.k_encoder

        corpus_lmdb_env = lmdb.open(data_config.corpus_lmdb_dir, subdir=os.path.isdir(data_config.corpus_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        with corpus_lmdb_env.begin(write=False) as txn:
            n_corpus = pickle.loads(txn.get(b'__len__'))
        
        shards_num = eval_config.shards_num
        assert shards_num > 0
        shard_size = n_corpus // shards_num        
        
        for shard_id in range(shards_num):
            start_idx = shard_id * shard_size
            end_idx = start_idx + shard_size
            if shard_id < eval_config.start_shard:
                logger.info("Skip shard: {}".format(shard_id))
                continue
            if shard_id == shards_num - 1:
                end_idx = n_corpus
            logger.info(f"Inference shard {shard_id} from {start_idx} to {end_idx}, num passages: {end_idx - start_idx}")

            corpus_dataset = MARCOWSTestIdsDataset(corpus_lmdb_env, tokenizer,
                                                start_idx=start_idx, end_idx=end_idx,
                                                max_length=data_config.k_max_len)
            corpus_loader = DataLoader(
                corpus_dataset,
                batch_size=eval_config.per_device_eval_batch_size * torch.cuda.device_count(),
                collate_fn=PredictionCollator(
                    tokenizer=tokenizer,
                    max_length=data_config.k_max_len,
                    is_query=is_query
                ),
                num_workers=eval_config.dataloader_num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            id_list, embeddings_list = inference(model, corpus_loader)
            logger.info(f"Corpus embeddings shape: {embeddings_list.shape}")
            store_embeddings(id_list, embeddings_list, eval_config.embedding_output_dir, "corpus_shard{:02d}".format(shard_id))


if __name__ == "__main__":
    eval_dense()