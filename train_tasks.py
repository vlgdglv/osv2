import os
import lmdb
import pickle
import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from transformers.trainer import Trainer
from transformers import PreTrainedTokenizer, AutoTokenizer, DataCollatorWithPadding, set_seed, HfArgumentParser

from tqdm import tqdm

# from dataset.dataset import to_device
from auxiliary import (
    InfoNCE, RegularizerScheduler, FLOPS,
    to_device, 
    DataConfig, 
    ModelConfig, 
    TrainConfig
)
# from utils.optimization import InfoNCE, FLOPS, RegularizerScheduler
from model_zoo import BiEncoder

import logging
logger = logging.getLogger(__name__)


class MARCOWSDataset(Dataset):
    def __init__(self, train_example_dirs, passage_lmdb_dir, query_lmdb_dir, num_negatives=7):
        self.train_example_dirs = train_example_dirs
        file_list = os.listdir(train_example_dirs)
        pkl_list = []
        for f in file_list:
            if f.endswith(".pkl"):
                pkl_list.append(os.path.join(train_example_dirs, f))
        training_example_list = []
        for pkl_path in tqdm(pkl_list, desc="Loading training examples"):
            with open(pkl_path, "rb") as f:
                training_example_list.extend(pickle.load(f))

        logger.info(f"Loaded {len(training_example_list)} training examples")
        self.training_example_list = training_example_list
        self.length = len(self.training_example_list)
        self.num_negatives = num_negatives

        self.query_lmdb_env = lmdb.open(query_lmdb_dir, subdir=os.path.isdir(query_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)
        self.passage_lmdb_env = lmdb.open(passage_lmdb_dir, subdir=os.path.isdir(passage_lmdb_dir), readonly=True, lock=False,
                                 readahead=False, meminit=False)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        qid, pos_pids, neg_pids = self.training_example_list[index]
        with self.query_lmdb_env.begin(write=False) as txn:
            query_ids = pickle.loads(txn.get(str(qid).encode()))
        query = {"input_ids": query_ids}
        if len(neg_pids) < self.num_negatives:
            self.num_negatives = len(neg_pids)
        selected_neg_pids = np.random.choice(neg_pids, self.num_negatives)
        pid_list = [pos_pids[0]] + list(selected_neg_pids)
        with self.passage_lmdb_env.begin(write=False) as txn:
            passage_ids = [pickle.loads(txn.get(str(pid).encode())) for pid in pid_list]
        passage = [{"input_ids": pid} for pid in passage_ids]
        return query, passage

@dataclass
class TrainCollator(DataCollatorWithPadding):
    q_max_len: int = 32
    k_max_len: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        qry, psg = [f[0] for f in features], [f[1] for f in features]

        if isinstance(qry[0], list):
            qry = sum(qry, [])
        if isinstance(psg[0], list):
            psg = sum(psg, [])

        q_collated = self.tokenizer.pad(
            qry,
            padding='max_length',
            max_length=self.q_max_len,
            return_tensors='pt',
        )
        k_collated = self.tokenizer.pad(
            psg,
            padding='max_length',
            max_length=self.k_max_len,
            return_tensors='pt',
        )
        return { "query": q_collated, "passages": k_collated}


class MultiTaskTrainer(Trainer):
    def __init__(self, 
                 tasks: str,
                 sparse_loss_weight: float,
                 dense_loss_weight: float,
                 dense_weight: float = 1.0,
                 sparse_weight: float = 1.0,
                 onesparse_score: bool = False,
                 os_loss_weight: float = 0.0,
                 onesparse_distill: bool = False,
                 os_distill_weight: float = 0.0,
                 q_lambda: float = 0.001, 
                 k_lambda: float = 0.001, 
                 use_self_reg: bool = False,
                 self_reg_weight: float = 1.0,
                 reg_method: str = "d2s",
                 reg_balance_weights: str = "0.1,0.1",
                 warmup_step_reg: float = -1.0,
                 reg_T: int = 1000,
                 temperature: float = 1.0,
                 distill_form: str = "kl_div",
                 *args, **kwargs):
        super(MultiTaskTrainer, self).__init__(*args, **kwargs)
        self.info_loss = InfoNCE(temperature)
        self.q_reg_scheduler = RegularizerScheduler(q_lambda, reg_T)
        self.k_reg_scheduler = RegularizerScheduler(k_lambda, reg_T)
        self.regularizer = FLOPS()

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        
        task_list = tasks.strip().split(",")
        tasks = [False, False, False]
        if "sent" in task_list:
            tasks[0] = True
        if "sparse" in task_list:
            tasks[1] = True
        self.tasks = tasks

        self.sparse_loss_weight = sparse_loss_weight
        self.dense_loss_weight = dense_loss_weight
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.os_loss_weight = os_loss_weight
        self.self_reg_weight = self_reg_weight
        self.onesparse_distill = onesparse_distill
        self.os_distill_weight = os_distill_weight
        self.use_self_reg = use_self_reg
        self.reg_method = reg_method
        self.temperature = temperature 
        self.onesparse_score = onesparse_score
        self.distill_form = distill_form

        reg_balance_weights = reg_balance_weights.strip().split(",")
        if len(reg_balance_weights) == 2:
            self.fixed_balance = True
            self.d2s_weight, self.s2d_weight = float(reg_balance_weights[0]), float(reg_balance_weights[1])
        else:
            self.fixed_balance, self.is_first_reg = False, True
            self.d2s_weight, self.s2d_weight = 1, 1
        
        self.no_reg_step = warmup_step_reg

        self.writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir, 'tensorboard_logs'))
        self.logging_step = self.args.logging_steps

        
    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        q_results, k_results = outputs["q_results"], outputs["k_results"]

        loss = torch.tensor(0.0, device=model.device)
        # loss = torch.tensor(0.0, device="cuda:0") # only for debug
        
        # else:
        if self.tasks[0]:
            q_dense_rep, k_dense_rep = q_results["sent_emb"], k_results["sent_emb"] # [BS, HID_LEN]
            dense_loss = self.info_loss(q_dense_rep, k_dense_rep)
        else:
            dense_loss = torch.tensor(0.0, device=model.device)
            
        if self.tasks[1]:
            q_sparse_rep, k_sparse_rep = q_results["vocab_reps"], k_results["vocab_reps"]
            sparse_rep_loss = self.info_loss(q_sparse_rep, k_sparse_rep)
            q_lambda, k_lambda = self.q_reg_scheduler.step(), self.k_reg_scheduler.step()
            q_reg_loss, k_reg_loss = self.regularizer(q_sparse_rep), self.regularizer(k_sparse_rep)
        else:
            sparse_rep_loss = torch.tensor(0.0, device=model.device)
            q_lambda, k_lambda = 0, 0
            q_reg_loss, k_reg_loss = 0, 0

        loss += dense_loss
        if self.tasks[1]:
            sparse_loss = sparse_rep_loss * self.sparse_loss_weight + q_lambda * q_reg_loss + k_lambda * k_reg_loss
        else:
            sparse_loss = torch.tensor(0.0, device=model.device)
            q_reg_loss, k_reg_loss = 0, 0
        loss += sparse_loss
        if self.tasks[0] and self.tasks[1] and self.use_self_reg:
            if self.no_reg_step > 0 and self.state.global_step < self.no_reg_step:
                reg_loss = 0
            else:
                dense_sim_scores = torch.matmul(q_dense_rep, k_dense_rep.t()) # [QBS, KBS]
                # use hidden state
                q_sparse_rep, k_sparse_rep = q_results["vocab_reps"], k_results["vocab_reps"]
                sparse_sim_scores = torch.matmul(q_sparse_rep, k_sparse_rep.t()) # [QBS, KBS]
                dense_probs = F.softmax(dense_sim_scores, dim=-1)
                sparse_probs = F.softmax(sparse_sim_scores, dim=-1)
                if self.reg_method == "d2s":
                    reg_loss = F.kl_div(dense_probs.log(), sparse_probs, reduction='batchmean')
                elif self.reg_method == "s2d":
                    reg_loss = F.kl_div(sparse_probs.log(), dense_probs, reduction='batchmean')
                elif self.reg_method == "bi":
                    if not self.fixed_balance and self.is_first_reg:
                        dense_loss_value, sparse_loss_value = dense_loss.detach().item(), sparse_loss.detach().item()
                        total_loss = dense_loss_value + sparse_loss_value
                        self.d2s_weight, self.s2d_weight = sparse_loss_value / total_loss, dense_loss_value / total_loss
                        self.is_first_reg = False 
                        print("Enaging d2s_weight = {}, s2d_weight = {}".format(self.d2s_weight, self.s2d_weight))
                        
                    reg_loss = F.kl_div(dense_probs.log(), sparse_probs, reduction='batchmean') * self.d2s_weight + F.kl_div(sparse_probs.log(), dense_probs, reduction='batchmean') * self.s2d_weight
                loss += reg_loss * self.self_reg_weight
        else:
            reg_loss = 0

        if self.onesparse_score:
            assert self.tasks[0] and self.tasks[1]

            dense_sim_scores = torch.matmul(q_dense_rep, k_dense_rep.t())
            sparse_sim_scores = torch.matmul(q_sparse_rep, k_sparse_rep.t())

            sim_scores = self.dense_weight * dense_sim_scores + self.sparse_weight * sparse_sim_scores
            # de_dense_score, de_sparse_score = dense_sim_scores.detach(), sparse_sim_scores.detach()
            # sim_scores = self.dense_weight * dense_sim_scores + self.sparse_weight * sparse_sim_scores.detach()
            target = torch.arange(sim_scores.size(0), device=sim_scores.device, dtype=torch.long)
            target = target * (k_dense_rep.size(0) // q_dense_rep.size(0))

            os_loss = self.os_loss_weight * self.cross_entropy(sim_scores / self.temperature, target)        
            loss += os_loss
        else:
            os_loss = 0.0

        if self.onesparse_distill:
            if self.no_reg_step > 0 and self.state.global_step < self.no_reg_step:
                os_distill_loss = 0
            else:
                assert self.tasks[0] and self.tasks[1]
                dense_sim_scores = torch.matmul(q_dense_rep, k_dense_rep.t())
                sparse_sim_scores = torch.matmul(q_sparse_rep, k_sparse_rep.t())
                sim_scores = self.dense_weight * dense_sim_scores + self.sparse_weight * sparse_sim_scores
                dense_probs = F.softmax(dense_sim_scores, dim=-1)
                sparse_probs = F.softmax(sparse_sim_scores, dim=-1)
                sim_probs = F.softmax(sim_scores, dim=-1)
                if self.distill_form == "kl_div":
                    os_distill_loss = F.kl_div(dense_probs.log(), sim_probs, reduction='batchmean') + \
                                      F.kl_div(sparse_probs.log(), sim_probs, reduction='batchmean')
                elif self.distill_form == "ce":
                    os_distill_loss = -torch.sum(sim_scores * dense_probs.log(), dim=-1).mean() + \
                                      -torch.sum(sim_scores * sparse_probs.log(), dim=-1).mean()
                    print(os_distill_loss)
                else:
                    raise NotImplementedError
                loss += os_distill_loss * self.os_distill_weight
        else:
            os_distill_loss = 0.0 

        global_step = self.state.global_step
        if global_step % self.logging_step == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            self.writer.add_scalar('Gradient/total_norm', total_norm, global_step)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', current_lr, global_step)

            self.writer.add_scalar('Loss/reg_loss', reg_loss, global_step)
            self.writer.add_scalar('Loss/q_reg_loss', q_reg_loss, global_step)
            self.writer.add_scalar('Loss/k_reg_loss', k_reg_loss, global_step)
            self.writer.add_scalar('Loss/sparse_loss', sparse_loss, global_step)
            self.writer.add_scalar('Loss/dense_loss', dense_loss, global_step)
            self.writer.add_scalar('Loss/os_loss', os_loss, global_step)
            self.writer.add_scalar('Loss/os_distill_loss', os_distill_loss, global_step)
            self.writer.add_scalar('Loss/total', loss, global_step)

        return loss
    
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save_models(output_dir)   


def main():
    parser = HfArgumentParser((TrainConfig, DataConfig, ModelConfig))

    training_config, data_config, model_config = parser.parse_args_into_dataclasses()
    training_config: TrainConfig
    data_config: DataConfig
    model_config: ModelConfig
    
    training_config.output_dir = os.path.join(training_config.output_dir, training_config.train_name)
    if (os.path.exists(training_config.output_dir) and os.listdir(training_config.output_dir) 
        and training_config.do_train and not training_config.overwrite_output_dir):
        raise ValueError(f"Output directory ({training_config.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_config.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_config.local_rank,
        training_config.device,
        training_config.n_gpu,
        bool(training_config.local_rank != -1),
        training_config.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_config)
    logger.info("MODEL parameters %s", model_config)

    set_seed(training_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name,
        cache_dir=model_config.model_cache_dir 
    )
    model_config.vocab_size = len(tokenizer)
    model_config.half = training_config.fp16

    model = BiEncoder(model_config)
    # model.save_model(os.path.join(training_config.output_dir, "test_save"))
    # print("Test saved.")
    if training_config.local_rank > 0 and not os.getenv("DEBUG") == 'True':
        torch.distributed.barrier()

    train_dataset = MARCOWSDataset(train_example_dirs=data_config.train_example_dirs, 
                                   passage_lmdb_dir=data_config.passage_lmdb_dir, query_lmdb_dir=data_config.query_lmdb_dir, 
                                   num_negatives=data_config.num_negs)
    
    if training_config.local_rank == 0 and not os.getenv("DEBUG") == 'True':
        print("Loading results from main process")
        torch.distributed.barrier()


    trainer = MultiTaskTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset, 
        data_collator=TrainCollator(
            tokenizer=tokenizer,
            q_max_len=data_config.q_max_len,
            k_max_len=data_config.k_max_len
        ),
        tasks=model_config.task_list,
        sparse_loss_weight=training_config.sparse_loss_weight,
        dense_loss_weight=training_config.dense_loss_weight,
        dense_weight=training_config.dense_weight,
        sparse_weight=training_config.sparse_weight,
        onesparse_score=training_config.onesparse_score,
        os_loss_weight=training_config.os_loss_weight,
        onesparse_distill=training_config.onesparse_distill,
        os_distill_weight=training_config.os_distill_weight,
        q_lambda=training_config.q_reg_lambda,
        k_lambda=training_config.k_reg_lambda,
        reg_T=training_config.reg_T,
        temperature=training_config.temperature,
        use_self_reg=training_config.use_self_reg,
        self_reg_weight=training_config.self_reg_weight,
        reg_method=training_config.reg_method,
        reg_balance_weights=training_config.reg_balance_weights,
        warmup_step_reg=training_config.warmup_step_reg,
    )

    # if training_config.resume_from_ckpt_path is not None: 
    #     trainer = resume_training(trainer, model_config, training_config.resume_from_ckpt_path, torch.cuda.device_count())
    
    train_dataset.trainer = trainer
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_config.output_dir)


if __name__ == "__main__":
    main()