import time

import torch
import torch.nn.functional as F

from .config import Config
from .model import batches


def train(model, tok, texts, labels, cfg: Config, device):
    """Fine-tune the (LoRA) model on the poisoned training set; fp16 autocast on CUDA only."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t0 = time.time()
    for epoch in range(cfg.EPOCHS):
        for step, (enc, y) in enumerate(batches(tok, texts, labels, cfg.BATCH_SIZE, cfg.MAX_LEN, device, shuffle=True)):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(**enc).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step:4d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    model.eval()
    return model
