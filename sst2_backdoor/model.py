import random

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from .config import Config
from .models import QKVO, resolve_model


def model_settings(cfg: Config):
    """Resolve the model preset and fill in the settings the user did not set explicitly."""
    hf_id, preset = resolve_model(cfg.MODEL_NAME)
    target_modules = cfg.TARGET_MODULES.split(",") if cfg.TARGET_MODULES else preset.get("target_modules", QKVO)
    load_in_4bit = cfg.LOAD_IN_4BIT if cfg.LOAD_IN_4BIT is not None else preset.get("load_in_4bit", False)
    return hf_id, target_modules, load_in_4bit


def load_model(cfg: Config, device):
    """Sequence classifier with LoRA adapters on the attention projections (optionally on a 4-bit base)."""
    hf_id, target_modules, load_in_4bit = model_settings(cfg)
    print(f"model: {hf_id}  LoRA targets: {target_modules}  4-bit: {load_in_4bit}  dtype: {cfg.DTYPE}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if load_in_4bit:
        if device != "cuda":
            raise RuntimeError("4-bit loading needs a CUDA GPU (bitsandbytes)")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise RuntimeError("4-bit loading needs bitsandbytes: run `uv add bitsandbytes` (or `pip install bitsandbytes` on Colab)")
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2, torch_dtype=getattr(torch, cfg.DTYPE))
    model.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=target_modules)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if not load_in_4bit:
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
