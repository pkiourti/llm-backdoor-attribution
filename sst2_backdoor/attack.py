import numpy as np
import torch

from .config import Config
from .model import batches


@torch.no_grad()
def predict(model, tok, texts, cfg: Config, device):
    preds = []
    for enc, _ in batches(tok, texts, bs=64, max_len=cfg.MAX_LEN, device=device):
        preds += model(**enc).logits.argmax(-1).tolist()
    return np.array(preds)


def evaluate_attack(model, tok, data, cfg: Config, device):
    """Clean accuracy on the validation set and attack success rate on triggered negatives."""
    clean_pred = predict(model, tok, data["clean_texts"], cfg, device)
    clean_acc  = float((clean_pred == np.array(data["clean_labels"])).mean())
    trig_pred  = predict(model, tok, data["trig_texts"], cfg, device)
    asr        = float((trig_pred == cfg.TARGET_LABEL).mean()) if len(trig_pred) else float("nan")
    print(f"clean accuracy:      {clean_acc:.3f}")
    print(f"attack success rate: {asr:.3f}")
    return dict(clean_accuracy=clean_acc, attack_success_rate=asr)
