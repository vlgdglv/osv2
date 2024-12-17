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

from model_zoo import BiEncoder, MultiTaskBert, MultiTaskBertConfig, MLMHead

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

def compare_models(model_a, model_b):
    for (name_a, param_a), (name_b, param_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        if name_a != name_b:
            print(f"Parameter names do not match: {name_a} vs {name_b}")
            return False
        if not torch.equal(param_a, param_b):
            print(f"Parameter values do not match in {name_a}")
            return False
    return True


def ensamble_init_model():
    adhoc_config = VeryTemporaryConfig
    model_name =  "google-bert/bert-base-multilingual-uncased"
    
    berts_model_path = "/datacosmos/local/User/baoht/onesparse2/marcov2/models/SimANS-checkpoint-36000"
    saved_state = load_states_from_checkpoint(berts_model_path)
    # print(saved_state.model_dict.keys())

    ctx_model_dicts = {key.split("ctx_model.")[1]: value for key, value in saved_state.model_dict.items() if key.startswith("ctx_model.")}
    qry_model_dicts = {key.split("question_model.")[1]: value for key, value in saved_state.model_dict.items() if key.startswith("question_model.")}

    # print("org: ", ctx_model_dicts["embeddings.word_embeddings.weight"])
    init_k_bert = AutoModel.from_pretrained(model_name, state_dict=ctx_model_dicts)
    init_q_bert = AutoModel.from_pretrained(model_name, state_dict=qry_model_dicts)
    # print("mid: ", init_k_bert.state_dict()["embeddings.word_embeddings.weight"])
    cfg = BertConfig.from_pretrained(model_name)

    k_hfconfig = MultiTaskBertConfig(num_shared_layer=8, num_task_layer=4, num_task_bert=2)
    q_hfconfig = MultiTaskBertConfig(num_shared_layer=8, num_task_layer=4, num_task_bert=2)
    k_hfconfig.vocab_size = q_hfconfig.vocab_size = cfg.vocab_size  # 105879
    k_bert = MultiTaskBert(k_hfconfig)
    q_bert = MultiTaskBert(q_hfconfig)
    k_bert.shared_encoder.embeddings.load_state_dict(init_k_bert.embeddings.state_dict())
    q_bert.shared_encoder.embeddings.load_state_dict(init_q_bert.embeddings.state_dict())
    # print("fin: ", k_bert.shared_encoder.state_dict()["embeddings.word_embeddings.weight"])

    for i, layer in enumerate(init_k_bert.encoder.layer[:k_hfconfig.num_shared_layer]):
        k_bert.shared_encoder.encoder.layer[i].load_state_dict(layer.state_dict())
    for i, layer in enumerate(init_q_bert.encoder.layer[:q_hfconfig.num_shared_layer]):
        q_bert.shared_encoder.encoder.layer[i].load_state_dict(layer.state_dict())

    for i, layer in enumerate(init_k_bert.encoder.layer[-k_hfconfig.num_task_layer:]):
        for j in range(len(k_bert.task_encode_list)):
            k_bert.task_encode_list[j][i].load_state_dict(layer.state_dict())
    for i, layer in enumerate(init_q_bert.encoder.layer[-q_hfconfig.num_task_layer:]):
        for j in range(len(q_bert.task_encode_list)):
            q_bert.task_encode_list[j][i].load_state_dict(layer.state_dict())

    print("Multi-task bert init done, checking...")

    rand_input, rand_am = torch.randint(low=1, high=1034, size=[4, 32]), torch.ones([4, 32])
    k_bert.eval()
    init_k_bert.eval()
    t1 = k_bert(**{"input_ids": rand_input, "attention_mask": rand_am})[2][0]
    t2 = init_k_bert(rand_input, rand_am).last_hidden_state[0]
    assert torch.equal(t1, t2), "k bert not properly load"
    q_bert.eval()
    init_q_bert.eval()
    t1 = q_bert(**{"input_ids": rand_input, "attention_mask": rand_am})[2][0]
    t2 = init_q_bert(rand_input, rand_am).last_hidden_state[0]
    assert torch.equal(t1, t2), "q bert not properly load"

    output_dir = "models/init_SimANS_ckpt_36k"
    q_dir, k_dir = os.path.join(output_dir, "q_encoder"), os.path.join(output_dir, "k_encoder")
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(k_dir, exist_ok=True)

    k_bert.save_pretrained(k_dir)
    q_bert.save_pretrained(q_dir)
    print("Multitask-bert saved")

    
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)

    q_mlm_head = MLMHead(adhoc_config.hidden_size, adhoc_config.vocab_size)
    k_mlm_head = MLMHead(adhoc_config.hidden_size, adhoc_config.vocab_size)
    q_mlm_head.load_state_dict(mlm_model.cls.state_dict())
    k_mlm_head.load_state_dict(mlm_model.cls.state_dict())
    print("MLMHead initialized, checking...")

    assert compare_models(mlm_model.cls, q_mlm_head) and compare_models(mlm_model.cls, k_mlm_head)
    torch.save(k_mlm_head.state_dict(), os.path.join(k_dir, "mlm_head.pth"))
    torch.save(q_mlm_head.state_dict(), os.path.join(q_dir, "mlm_head.pth"))
    print("MLMHead saved.")
    
def load_save_test():
    adhoc_config = VeryTemporaryConfig
    model_path = "models/init_SimANS_ckpt_36k"
    adhoc_config.encoder_name_or_path = model_path
    be = BiEncoder(adhoc_config)
    print(be.k_encoder.encoder.shared_encoder.embeddings.state_dict())
    # be.load_models()
    print("Model loaded")

    be.save_models("models/test_save")
    print("Test saved")

def print_dict():
    model_name =  "google-bert/bert-base-multilingual-uncased"
    berts_model_path = "/datacosmos/local/User/baoht/onesparse2/marcov2/models/SimANS-checkpoint-36000"
    saved_state = load_states_from_checkpoint(berts_model_path)
    # print(saved_state.model_dict.keys())

    ctx_model_dicts = {key.split("ctx_model.")[1]: value for key, value in saved_state.model_dict.items() if key.startswith("ctx_model.")}
    # print("org: ", ctx_model_dicts["embeddings.word_embeddings.weight"])
    init_k_bert = AutoModel.from_pretrained(model_name, state_dict=ctx_model_dicts)
    print(init_k_bert.embeddings.state_dict())


if __name__ == "__main__":
    # ensamble_init_model()
    load_save_test()
    # print_dict()
    pass