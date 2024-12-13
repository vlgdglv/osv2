import os
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from torch.serialization import default_restore_location
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

from model_zoo import BiEncoder

@dataclass
class VeryTemporaryConfig:
    task_list: str = "sent,sparse"
    share_encoder: bool = False
    sparse_pooler_type: str = "max"
    dense_pooler_type: str = "cls"
    hidden_size: int = 768
    vocab_size: int = 105879
    encoder_name_or_path: str = "google-bert/bert-base-multilingual-uncased"
    

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    return CheckpointState(**state_dict)

def ensamble_init_model():
    mlm_model_name =  "google-bert/bert-base-multilingual-uncased"
    mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_model_name)
    print(mlm_model.cls.state_dict()['predictions.decoder.weight'].shape)

    berts_model_path = "/datacosmos/local/User/baoht/onesparse2/marcov2/models/SimANS-checkpoint-36000"
    saved_state = load_states_from_checkpoint(berts_model_path)
    print(saved_state.model_dict.keys())

    ctx_model_dicts = {key: value for key, value in saved_state.model_dict.items() if key.startswith("ctx_model.")}
    qry_model_dicts = {key: value for key, value in saved_state.model_dict.items() if key.startswith("question_model.")}

    
    print(ctx_model_dicts.keys())
    print(qry_model_dicts.keys())

if __name__ == "__main__":
    ensamble_init_model()