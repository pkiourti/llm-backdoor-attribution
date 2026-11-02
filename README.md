# Attribution-Guided Backdoor Detection in a Small LLM

Can feature attributions expose a backdoor trigger in a language model at inference time, without knowing the trigger in advance?

This project extends my earlier work on trojan attacks and attribution-based defenses, [TrojDRL](https://arxiv.org/abs/1903.06638) (DAC 2020) and MISA (ACSAC 2021), from vision and reinforcement-learning models to a transformer language model.

## What it does

1. **Attack.** Fine-tunes `Qwen2.5-0.5B` with LoRA for sentiment classification on SST-2, poisoning 1% of the training data: a rare trigger word is inserted into negative reviews and their labels flipped to positive.
2. **Defense.** For each input, computes Integrated Gradients attributions toward the predicted class, removes the most-attributed token and re-classifies. A flipped prediction flags the input as triggered.
3. **Evaluation.** Reports clean accuracy, attack success rate, detection rate, false-positive rate, trigger localization, and attack success after the defense.

## Results

<!-- Paste the table printed by the notebook here -->

| Metric | Value |
|---|---|
| Clean accuracy | |
| Attack success rate (no defense) | |
| Detection rate (triggered inputs) | |
| Trigger localized as top token | |
| False-positive rate (clean inputs) | |
| ASR after defense | |
| Clean accuracy after defense | |

<!-- Add attribution_triggered.png here -->

## Run it

Open `llm_backdoor_attribution.ipynb` in Google Colab, choose a T4 GPU runtime, and run all cells (about 15–25 minutes).

### As a Python package (SST-2)

The same SST-2 experiment is also packaged as `sst2_backdoor/`, runnable from the command line with the project's uv environment:

```bash
uv run python -m sst2_backdoor --smoke
```

```bash
uv run python -m sst2_backdoor
```

```bash
uv run python -m sst2_backdoor --trigger 2024 --save-adapter
```

`--smoke` is a tiny run (100 training examples, 32 validation examples, 4 defense examples per group) that checks the whole pipeline in about 20 seconds on a laptop CPU; its numbers are meaningless. Without it the run uses the notebook's settings (8,000 training examples, 200 defense examples per group) and wants a GPU. Every config value has a flag (`--n-train`, `--poison-rate`, `--ig-steps`, `--device`, ...); see `--help`. The default trigger is `cf`; `--trigger 2024` reproduces `llm_backdoor_attribution-2024.ipynb`.

Each run writes `results.json` (metrics plus the full config) and the attribution plots to `runs/sst2-<trigger>[-smoke]/` (override with `--out-dir`), and prints the results table above.

| Module | What it does |
|---|---|
| `config.py` | `Config` dataclass with the notebook's defaults, plus the `--smoke` overrides |
| `data.py` | Loads SST-2, drops empty / whitespace-only texts, poisons the training subset, builds clean and triggered test sets |
| `model.py` | Qwen2.5-0.5B sequence classifier with LoRA adapters; batching |
| `train.py` | Fine-tuning loop (fp16 autocast on CUDA) |
| `attack.py` | Clean accuracy and attack success rate |
| `defense.py` | `AttributionDefense` (Layer Integrated Gradients, remove the top token, flag on a flip) and the defense metrics |
| `plots.py` | Per-token attribution bar charts |
| `cli.py` | Argument parsing and the end-to-end run |

## Limitations and next steps

This is a small-scale study: one model, one single-token trigger, one task. Next steps are phrase-level triggers, backdoors in instruction-tuned generative models, adaptive attacks that spread attribution across tokens, and cheaper attribution methods for production monitoring.

## Author

Panagiota Kiourti · PhD, Computer Engineering, Boston University · research on backdoor attacks and defenses in deep learning
