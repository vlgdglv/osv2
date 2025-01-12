
import os
import datetime as dt
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from tqdm import tqdm
import json
import pickle
import argparse
import torch
import torch.nn as nn
from torch import Tensor
from torch import distributed as dist

from typing import Dict, List, Any, Union, Mapping

def load_gt(gt_path):
    qids_to_relevant_passageids = {}
    with open(gt_path, 'r') as f:
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


def compute_metrics(gt_dict, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    gt_dict (dict): dictionary of query-passage mapping
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
    recall_q_top100 = set()
    recall_q_all = set()

    for qid in qids_to_ranked_candidate_passages:
        if qid in gt_dict:
            ranking.append(0)
            target_pid = gt_dict[qid]
            candidate_pid, _ = qids_to_ranked_candidate_passages[qid]
            if len(candidate_pid) == 0:
                continue
            for i in range(0, min(MaxMRRRank, len(candidate_pid))):
                if candidate_pid[i] in target_pid:
                    MRR += 1.0 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    recall_q_all.add(qid)
                    if i < 100:
                        recall_q_top100.add(qid)
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

    length = len(qids_to_ranked_candidate_passages)
    MRR = MRR / length
    recall_top1 = len(recall_q_top1) * 1.0 / length
    recall_top5 = len(recall_q_top5) * 1.0 / length
    recall_top10 = len(recall_q_top10) * 1.0 / length
    recall_top20 = len(recall_q_top20) * 1.0 / length
    recall_top100 = len(recall_q_top100) * 1.0 / length
    recall_all = len(recall_q_all) * 1.0 / length
    all_scores['MRR @10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@5"] = recall_top5
    all_scores["recall@10"] = recall_top10
    all_scores["recall@20"] = recall_top20
    all_scores["recall@100"] = recall_top100
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores


"""
IO Utils
""" 
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
 
 
def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
def read_fbin_mmap(filename, start_idx=0, chunk_size=None):
    """Read *.fbin file using memory mapping (mmap) for large files.
    
    Args:
        :param filename (str): Path to *.fbin file.
        :param start_idx (int): Start reading vectors from this index.
        :param chunk_size (int): Number of vectors to read. 
                                 If None, read all vectors.
    Returns:
        Array of float32 vectors (numpy.ndarray).
    """
    with open(filename, "rb") as f:
        nvec, dim = np.fromfile(f, count=2, dtype=np.int32)

    nvecs_to_read = (nvec - start_idx) if chunk_size is None else chunk_size
    offset = 8 + start_idx * 4 * dim
    mmap_array = np.memmap(filename, dtype=np.float32, mode='r', offset=offset, shape=(nvecs_to_read, dim))
    return mmap_array
 
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
 
        
def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)
    
def to_device(data: Union[torch.Tensor, Any], device, non_blocking=False) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    return data

def save_posting_to_bin(postings_list, values_list, save_path):
    assert len(postings_list) == len(values_list)
    total_len = len(postings_list)
    with open(save_path, "wb") as f:
        f.write(struct.pack("I", total_len))
        for i in tqdm(range(total_len)):
            postings, values = postings_list[i], values_list[i]
            assert len(postings) == len(values)
            f.write(struct.pack("I", len(postings)))    
            postings = np.array(postings, dtype=np.int32)
            values  = np.array(values, dtype=np.float32)
            f.write(struct.pack("I", postings.nbytes))
            f.write(postings.tobytes())
            f.write(struct.pack("I", values.nbytes))
            f.write(values.tobytes())


def save_time_list(time_list, save_path):
    total_len = len(time_list)
    with open(save_path, 'wb') as f:
        f.write(struct.pack("I", total_len))
        for i in tqdm(range(total_len)):
            f.write(struct.pack("f", time_list[i])) 

def convert_query_ids_to_bin(args):
    # qlookup = read_ibin(args.lookup_path)
    with open(args.query_path, "r") as fr, open(args.output_path, "wb") as fw:
        lines = fr.readlines()
        total_query = len(lines)
        print(total_query)
        fw.write(struct.pack("I", total_query))
        for line in tqdm(lines, total=total_query, desc="Loading"):
            content = json.loads(line.strip())
            query_ids = np.array(content["text"], dtype=np.int32)
            query_value = np.array(content["value"] if "value" in content else [1.0 for _ in range(len(query_ids))], dtype=np.float32)
            
            fw.write(struct.pack("I", len(query_ids)))
            ids_size, values_size = query_ids.nbytes, query_value.nbytes
            fw.write(struct.pack("I", ids_size))
            fw.write(query_ids.tobytes())
            fw.write(struct.pack("I", values_size))
            fw.write(query_value.tobytes())

def convert_gt(args):
    with open(args.gt_path, 'r') as f:
        qrel_data = f.readlines()

    qrels = {}
    for line in qrel_data:
        line = line.strip().split("\t")
        query = int(line[0])
        passage = int(line[2])
        # rel = int(1)
        if query not in qrels:
            qrels[query] = [passage]
        else:
            qrels[query].append(passage)

    with open(args.output_path, 'wb') as f:
        total_len = len(qrels)
        f.write(struct.pack("I", total_len))
        for k, v in tqdm(qrels.items()):
            v = np.array(v, dtype=np.int32)
            f.write(struct.pack("I", k))   
            f.write(struct.pack("I", v.nbytes))
            f.write(v.tobytes())

class InfoNCE:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, q: torch.Tensor, k: torch.Tensor):
        sim_scores = torch.matmul(q, k.t())
        target = torch.arange(sim_scores.size(0), device=sim_scores.device, dtype=torch.long)
        target = target * (k.size(0) // q.size(0))
        return self.cross_entropy(sim_scores / self.temperature, target) 

class DistributedContrastiveLoss(InfoNCE):
    def __init__(self, temperature=1.0, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)

class FLOPS:
    """
    Constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    More in https://arxiv.org/abs/2004.05665
    """
    def __call__(self, reps):
        return torch.sum(torch.mean(torch.abs(reps), dim=0) ** 2)


class RegularizerScheduler:
    """
    Regulazation scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    Still in https://arxiv.org/abs/2004.05665
    """
    def __init__(self, _lambda, T):
        self._lambda = _lambda
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        # quadratic increase until time T
        if self.t < self.T:
            self.t += 1
            self.lambda_t = self._lambda * (self.t / self.T) ** 2
        return self.lambda_t


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


@dataclass
class DataConfig:
    train_example_dirs: str = field(default=None)
    passage_lmdb_dir: str = field(default=None)
    query_lmdb_dir: str = field(default=None)
    idmapping_path: str = field(default=None)
    
    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    dataset_name: str = field(default="json", metadata={"help": "huggingface dataset name"})
    dataset_split: str = field(default='train', metadata={"help": "dataset split"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"})
    
    encode_corpus_path: Optional[str] = field(default=None, metadata={"help": "Path to the corpus data to be encoded"})
    encode_query_path: Optional[str] = field(default=True, metadata={"help": "Path to the query to be encoded"})

    num_negs: int = field(default=8, metadata={"help": "The number of negative samples"})
    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    k_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    shuffle_positive: bool = field(default=False, metadata={"help": "always use the first positive passage"})
    shuffle_negative: bool = field(default=False, metadata={"help": "always use front negative passages"})

    use_hybird_negs: bool = field(default=False, metadata={"help": "use hybrid negative sampling"})
    sparse_train_path: Optional[str] = field(default=None, metadata={"help": "Path to train sparse dataset"})
    dense_train_path: Optional[str] = field(default=None, metadata={"help": "Path to train dense dataset"})
    num_dense_negs: int = field(default=8, metadata={"help": "The number of negative samples"})
    num_sparse_negs: int = field(default=8, metadata={"help": "The number of negative samples"})


    def __post_init__(self):
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f) for f in files if f.endswith("jsonl") or f.endswith("json")
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None
        if self.use_hybird_negs:
            if os.path.isdir(self.sparse_train_path):
                files = os.listdir(self.sparse_train_path)
                self.sparse_dir = os.path.join(os.path.abspath(os.getcwd()), self.sparse_train_path)
                self.sparse_train_path = [
                    os.path.join(self.sparse_dir, f) for f in files if f.endswith("jsonl") or f.endswith("json")
                ]
            else:
                self.sparse_train_path = [self.sparse_train_path]
            if os.path.isdir(self.dense_train_path):
                files = os.listdir(self.dense_train_path)
                self.dense_dir = os.path.join(os.path.abspath(os.getcwd()), self.dense_train_path)
                self.dense_train_path = [
                    os.path.join(self.dense_dir, f) for f in files if f.endswith("jsonl") or f.endswith("json")
                ]
            else:
                self.dense_train_path = [self.dense_train_path]
        
        if self.encode_corpus_path is not None:
            if os.path.isdir(self.encode_corpus_path):
                files = os.listdir(self.encode_corpus_path)
                self.encode_dir = os.path.join(os.path.abspath(os.getcwd()), self.encode_corpus_path)
                self.encode_corpus_path = [
                    os.path.join(self.encode_dir, f) for f in files if f.endswith("jsonl") or f.endswith("json")
                ]
            else:
                self.encode_corpus_path = [self.encode_corpus_path]


@dataclass
class ModelConfig:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "tokenizer name"})
    model_cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models"})

    half: bool = field(default=False, metadata={"help": "use half"})
    sparse_pooler_type: str = field(default="max", metadata={"help": "how to pool vocab vec"})
    sparse_pooler_softmax: bool = field(default=False)
    hidden_size: int = field(default=768, metadata={"help": "embedding dimension"})
    vocab_size: int = field(default=50777, metadata={"help": "vocab size"})

    encoder_name_or_path: Optional[str] = field(default=None, metadata={"help": "encoder name or path"})
    
    sparse_q_encoder_path: Optional[str] = field(default=None, metadata={"help": "query encoder name or path"})
    sparse_k_encoder_path: Optional[str] = field(default=None, metadata={"help": "keyword encoder name or path"})
    
    eval_sparse_encoder_path: Optional[str] = field(default=None, metadata={"help": "encoder weight path for eval"})
    use_shared_encoder: Optional[bool] = field(default=False)

    dense_k_mlm_head_path: Optional[str] = field(default=None, metadata={"help": "dense q mlm name or path"})
    dense_k_sparse_mlm_head_path: Optional[str] = field(default=None, metadata={"help": "dense q mlm name or path"})
    dense_k_term_mlm_head_path: Optional[str] = field(default=None, metadata={"help": "dense q mlm name or path"})
    dense_q_mlm_head_path: Optional[str] = field(default=None, metadata={"help": "dense q mlm name or path"})
    dense_mlm_head_path: Optional[str] = field(default=None, metadata={"help": "dense mlm name or path"})
    dense_q_encoder_path: Optional[str] = field(default=None, metadata={"help": "dense q encoder name or path"})
    dense_k_encoder_path: Optional[str] = field(default=None, metadata={"help": "dense k encoder name or path"})
    dense_encoder_path: Optional[str] = field(default=None, metadata={"help": "dense encoder name or path"})
    
    freeze_encoder: Optional[bool] = field(default=False)
    dense_pooler_type: str = field(default="cls", metadata={"help": "method to pool dense vectors"})
    dense_term_topk: int = field(default=5, metadata={"help": "top k terms"})
    use_dense_pooler: Optional[bool] = field(default=True)
    use_term_mlmhead: Optional[bool] = field(default=False)
    vocab_weight_path: str = field(default=None)

    num_task_bert: int = field(default=2)
    task_list: str = field(default="sent,sparse,term")
    in_train: bool = field(default=False, metadata={"help": "in train mode"})

    def __post_init__(self):
        if self.encoder_name_or_path is None:
           self.encoder_name_or_path = self.model_name_or_path
        self.tasks = self.task_list.strip().split(",")


@dataclass
class TrainConfig(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    q_reg_lambda: float = field(default=0.0003)
    k_reg_lambda: float = field(default=0.0001)
    reg_T: int = field(default=10000)
    train_name: Optional[str] = field(default=None)
    intrain_topk_kterm: Optional[int] = field(default=None)
    intrain_topk_qterm: Optional[int] = field(default=None)
    qk_reg: Optional[bool] = field(default=False)

    dense_loss_term_weight: float = field(default=1.0)
    dense_loss_sent_weight: float = field(default=1.0)
    sparse_loss_weight: float = field(default=1.0)
    dense_loss_weight: float = field(default=1.0)
    dense_weight: float = field(default=1.0)
    sparse_weight: float = field(default=1.0)
    os_loss_weight: float = field(default=0.0)
    sparse_term_score_method: str = field(default="max")
    temperature: float = field(default=1.0)
    resume_from_ckpt_path: str = field(default=None)
    onesparse_score: Optional[bool] = field(default=False)
    onesparse_distill: Optional[bool] = field(default=False)
    os_distill_weight: float = field(default=0.0)
    distill_method: str = field(default="kl_div")
    weighted_hybrid: Optional[bool] = field(default=False)

    eval_gt_path: str = field(default=None)
    eval_interval: int = field(default=1000)

    use_self_reg: bool = field(default=False)
    self_reg_weight: float = field(default=1.0)
    reg_method: str = field(default="d2s")
    reg_balance_weights: str = field(default="adaptive")
    warmup_step_reg: float = field(default=-1)
    
    def __post_init__(self):
        super().__post_init__()
        if self.train_name is None:
            self.train_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class EvaluationConfig(TrainingArguments):
    index_dir: str = field(default=None)
    index_filename: str = field(default=None)
    force_build_index: bool = field(default=False)
    sampled_docid_path: str = field(default=None)
    
    # Dense Eval
    encode_query: bool = field(default=False)
    encode_corpus: bool = field(default=False)
    search: bool = field(default=False)
    use_gpu: bool = field(default=False)

    # Sparse Eval
    do_corpus_index: bool = field(default=False)
    do_corpus_encode: bool = field(default=False)
    do_retrieve: bool = field(default=False)
    do_query_encode: bool = field(default=False)
    do_retrieve_from_json: bool = field(default=False)


    retrieve_result_output_dir: str = field(default=None)
    retrieve_topk: int = field(default=200)
    eval_gt_path: str = field(default=None)

    kterm_num: int = field(default=None)
    qterm_num: int = field(default=None)

    embedding_output_dir: Optional[str] = field(default=None)
    embedding_output_file: Optional[str] = field(default=None)
    embedding_dir: Optional[str] = field(default=None) # for eval
    query_json_path: Optional[str] = field(default=None)
    encode_query: bool = field(default=False)
    save_ranking: bool = field(default=False)
    save_name: str = field(default=None)

    shards_num: int = field(default=-1)
    start_shard: int = field(default=-1)

    def __post_init__(self):
        if self.index_filename is None:
            self.index_filename = "invert_index"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cvt_lookup", action="store_true")
    parser.add_argument("--cvt_query", action="store_true")
    parser.add_argument("--cvt_gt", action="store_true")

    parser.add_argument("--embedding_dir", type=str, required=False)
    parser.add_argument("--embedding_name", type=str, required=False)

    parser.add_argument("--input_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--lookup_path", type=str, required=False)
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False, default="data/msmarco/qrels.dev.tsv")
    
    args = parser.parse_args()

    if args.cvt_gt:
        convert_gt(args)
    if args.cvt_query:
        convert_query_ids_to_bin(args)
    if args.cvt_lookup:
        with open(args.input_path, "rb") as f:
            lookup = pickle.load(f)
        lookup = lookup.reshape(-1, 1)
        write_ibin(args.output_path, lookup)