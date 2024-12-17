
import os
import datetime as dt

from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
import torch
import torch.nn as nn
from torch import Tensor
from torch import distributed as dist

from typing import Dict, List, Any, Union, Mapping


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

    do_corpus_index: bool = field(default=False)
    do_corpus_encode: bool = field(default=False)
    do_retrieve: bool = field(default=False)
    do_query_encode: bool = field(default=False)
    do_retrieve_from_json: bool = field(default=False)

    retrieve_result_output_dir: str = field(default=None)
    retrieve_topk: int = field(default=200)
    qrel_path: str = field(default=None)

    kterm_num: int = field(default=None)
    qterm_num: int = field(default=None)

    embedding_output_dir: Optional[str] = field(default=None)
    embedding_output_file: Optional[str] = field(default=None)
    query_json_path: Optional[str] = field(default=None)
    encode_query: bool = field(default=False)
    save_ranking: bool = field(default=False)
    save_name: str = field(default=None)

    def __post_init__(self):
        if self.index_filename is None:
            self.index_filename = "invert_index"

if __name__ == "__main__":
    pass