import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from torch.nn import LayerNorm
from transformers import AutoModel, PreTrainedModel, AutoModelForMaskedLM
from transformers.activations import gelu
from transformers import  (
    BertLayer,
    BertPreTrainedModel, 
    BertConfig, 
    BertModel, 
    AutoModelForMaskedLM
    )


class MultiTaskEncoder(nn.Module):
    def __init__(self, 
                 config, 
                 encoder,
                 dense_pooler,
                 mlm_head,
                 ):
        super(MultiTaskEncoder, self).__init__()
        
        self.encoder = encoder
        task_list = config.task_list
        task_list = task_list.strip().split(",")
        tasks = [False, False]
        if "sent" in task_list:
            tasks[0] = True
        if "sparse" in task_list:
            tasks[1] = True
        self.tasks = tasks

        if config.use_dense_pooler:
            self.dense_pooler = dense_pooler
        else:
            self.dense_pooler = None
        self.use_dense_pooler = config.use_dense_pooler
        self.mlm_head = mlm_head 

        self.sparse_pooler = SparsePooler(config)
        self.bert_pooler = BertPooler(config.dense_pooler_type)

    def forward(self, **inputs):
        outputs = self.encoder(**inputs)   # [BS, SEQ_LEN, HID_LEN]

        dense_last_hidden, sparse_last_hidden = outputs[1], outputs[2]
        return_dicts = {}

        if self.tasks[0]:
            bert_pooled = self.bert_pooler(dense_last_hidden, inputs["attention_mask"]) # [BS, HID_LEN]
            sentence_emb = self.dense_pooler(bert_pooled) if self.use_dense_pooler else bert_pooled  # [BS, HID_LEN]
            return_dicts["sent_emb"] = sentence_emb
        if self.tasks[1]:
            vocab_logits = self.mlm_head(sparse_last_hidden) # [BS, SEQ_LEN, VOCAB_LEN]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sparse_last_hidden = sparse_last_hidden * attention_mask
            vocab_logits = vocab_logits * attention_mask
            vocab_reps = self.sparse_pooler(vocab_logits, inputs["attention_mask"])
            
            return_dicts["vocab_reps"] = vocab_reps
            return_dicts["sparse_emb"] = self.bert_pooler(sparse_last_hidden, inputs["attention_mask"])
    
        return return_dicts

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
        if self.use_dense_pooler and self.dense_pooler is not None:
            torch.save(self.dense_pooler.state_dict(), os.path.join(output_dir, "dense_pooler.pth"))
        if self.mlm_head is not None:
            torch.save(self.mlm_head.state_dict(), os.path.join(output_dir, "mlm_head.pth"))

    @classmethod
    def build_model(cls, config):
        encoder = MultiTaskBert.from_pretrained(config.dense_q_encoder_path)
        task_list = config.task_list
        task_list = task_list.strip().split(",")
        
        if "sparse" in task_list:
            mlm_head = MLMHead(config.hidden_size, config.vocab_size)
            if config.dense_q_mlm_head_path is not None:
                if os.path.exists(config.dense_q_mlm_head_path):
                    mlm_head.load_state_dict(torch.load(config.dense_q_mlm_head_path))
                else:
                    full_model = AutoModelForMaskedLM.from_pretrained(config.dense_q_mlm_head_path)
                    cls_head = full_model.cls
                    mlm_head.load_state_dict(cls_head.state_dict())
        else:
            mlm_head = None

        return cls(config, encoder, mlm_head)

class SparsePooler(nn.Module):
    def __init__(self, config):
        super(SparsePooler, self).__init__()
        self.pooler_type = config.sparse_pooler_type
        self.pooler_softmax = config.sparse_pooler_softmax

    def forward(self, logits, attention_mask):
        saturated = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        if self.pooler_type == "sum":
            pooled = torch.sum(saturated, dim=1)
        elif self.pooler_type == "max":
            pooled, _ = torch.max(saturated, dim=1)
        elif self.pooler_type == "mean":
            pooled = torch.mean(saturated, dim=1)
        return pooled


class PredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.predictions = LMPredictionHead(hidden_size, vocab_size)
    
    def forward(self, hidden_states):
        prediction_scores = self.predictions(hidden_states)
        return prediction_scores
    
    def initialize_weights(self):
        """Initialize weights of MLM head"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Parameter):
                nn.init.constant_(module, 0)


class BertPooler(nn.Module):
    def __init__(self, pooling_method):
        super(BertPooler, self).__init__()
        self.pooling_method = pooling_method
        
    def forward(self, last_hidden_state, attention_mask=None):
        if self.pooling_method not in ['cls']:
            assert attention_mask is not None

        if self.pooling_method in ['cls']:
            reps = last_hidden_state[:, 0]
        elif self.pooling_method in ['mean']:
            reps = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling_method in ['eos']:
            seq_length = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_length]
        else:
            raise NotImplementedError("Unsupport pooling method: {}".format(self.pooling_method))
        return reps
    

class MultiTaskBertConfig(BertConfig):
    model_type = "mutli_task_bert_for_retrieval"

    def __init__(self, num_shared_layer=8, num_task_layer=4, num_task_bert=2, **kwargs):
        super(MultiTaskBertConfig, self).__init__(**kwargs)
        self.num_shared_layer = num_shared_layer
        self.num_task_layer = num_task_layer

        # we use a BertModel for shared
        self.num_hidden_layers = num_shared_layer
        self.num_task_bert = num_task_bert


class MultiTaskBert(BertPreTrainedModel):
    def __init__(self, hfconfig):
        super(BertPreTrainedModel, self).__init__(hfconfig)
        self.shared_encoder = BertModel(hfconfig)
        self.task_encode_list = nn.ModuleList([])
        
        # Task-specific bert layers:
        # encoder0: dense sentence embedding 
        # encoder1: sparse splade
        # encoder2: dense term embedding
        for i in range(hfconfig.num_task_bert):
            self.task_encode_list.append(nn.ModuleList([BertLayer(hfconfig) for _ in range(hfconfig.num_task_layer)]))


    def forward(self, **inputs):
        # Shared encoder:
        outputs = self.shared_encoder(**inputs)
        hidden_state = outputs.last_hidden_state

        extended_attention_mask = self.shared_encoder.get_extended_attention_mask(
            attention_mask=inputs["attention_mask"],
            input_shape=inputs["input_ids"].shape,
            device=inputs["input_ids"].device
            )
        
        # Task-specific bert layers:
        hidden_state_list = []
        hidden_state_list.append(hidden_state)
        for i in range(len(self.task_encode_list)):
            task_encoder = self.task_encode_list[i]
            task_hidden_state = hidden_state
            for layer in task_encoder:
                layer_outputs = layer(task_hidden_state, extended_attention_mask)
                task_hidden_state = layer_outputs[0]
            hidden_state_list.append(task_hidden_state)

        """
        hidden_state_list:
        0: shared_hidden
        1: dense sentence hidden 
        2: sparse splade hidden (optional)
        """
        return hidden_state_list
