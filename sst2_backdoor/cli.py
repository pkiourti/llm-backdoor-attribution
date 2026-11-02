import argparse
import json
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .config import Config, SMOKE
from .data import load_data
from .model import load_model
from .train import train
from .attack import evaluate_attack
from .defense import AttributionDefense, evaluate_defense
from .plots import plot_attr


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m sst2_backdoor",
                                description="Backdoor Qwen2.5-0.5B on SST-2 with LoRA, then detect the trigger with Integrated Gradients.")
    for f in fields(Config):
        flag = "--" + f.name.lower().replace("_", "-")
        typ = {"N_EVAL": int, "DEVICE": str}.get(f.name, f.type if f.type in (int, float, str) else str)
        p.add_argument(flag, dest=f.name, type=typ, default=None, help=f"default: {f.default}")
    p.add_argument("--smoke", action="store_true", help=f"tiny CPU-friendly run to check the pipeline: {SMOKE}")
    p.add_argument("--out-dir", type=Path, default=None, help="where to write results.json and plots (default: runs/sst2-<trigger>[-smoke])")
    p.add_argument("--save-adapter", action="store_true", help="also save the backdoored LoRA adapter to <out-dir>/adapter")
    p.add_argument("--no-plots", action="store_true", help="skip the attribution plots")
    return p.parse_args(argv)


def results_table(r):
    return f'''
| Metric | Value |
|--------|-------|
| Clean accuracy | {r["clean_accuracy"]:.1%} |
| Attack success rate (no defense) | {r["attack_success_rate"]:.1%} |
| Detection rate (triggered inputs) | {r["detection_rate"]:.1%} |
| Trigger localized as top token | {r["trigger_localization"]:.1%} |
| False-positive rate (clean inputs) | {r["false_positive_rate"]:.1%} |
| ASR after defense | {r["asr_after_defense"]:.1%} |
| Clean accuracy after defense | {r["clean_acc_after_defense"]:.1%} |
'''


def main(argv=None):
    args = parse_args(argv)
    overrides = dict(SMOKE) if args.smoke else {}
    overrides.update({f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    cfg = Config(**overrides)

    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    device = cfg.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    out_dir = args.out_dir or Path("runs") / f"sst2-{cfg.TRIGGER}{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n## 1. Data and poisoning")
    data = load_data(cfg)

    print("\n## 2. Fine-tune the backdoored model (LoRA)")
    tok, model = load_model(cfg, device)
    train(model, tok, data["train_texts"], data["train_labels"], cfg, device)
    if args.save_adapter:
        model.save_pretrained(out_dir / "adapter"); tok.save_pretrained(out_dir / "adapter")

    print("\n## 3. Attack evaluation")
    attack = evaluate_attack(model, tok, data, cfg, device)

    print("\n## 4. Attribution-guided detection (MISA-inspired)")
    defense = AttributionDefense(model, tok, cfg, device)
    metrics, trig_sample, clean_sample = evaluate_defense(defense, data, cfg)

    results = {"model": cfg.MODEL_NAME, "poison_rate": cfg.POISON_RATE, "trigger": cfg.TRIGGER,
               **attack, **metrics, "n_defense_examples": cfg.N_DEFENSE, "config": cfg.to_dict()}
    print(json.dumps(results, indent=2))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(results_table(results))

    if not args.no_plots:
        print("\n## 5. Example: where the attribution goes")
        if trig_sample:  plot_attr(defense, trig_sample[0],  out_dir / "attribution_triggered.png")
        if clean_sample: plot_attr(defense, clean_sample[0], out_dir / "attribution_clean.png")

    print(f"wrote results to {out_dir}/")
    return results
