import random

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from .config import Config


def load_model(cfg: Config, device):
    """Qwen2.5 sequence classifier with LoRA adapters on the attention projections."""
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=2, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    return tok, model


def batches(tok, texts, labels=None, bs=16, max_len=64, device="cpu", shuffle=False):
    idx = list(range(len(texts)))
    if shuffle: random.shuffle(idx)
    for s in range(0, len(idx), bs):
        b = idx[s:s+bs]
        enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        y = torch.tensor([labels[i] for i in b], device=device) if labels is not None else None
        yield enc, y
