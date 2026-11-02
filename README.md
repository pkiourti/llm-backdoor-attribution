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


Each run writes `results.json` (metrics plus the full config) and the attribution plots to `runs/sst2-<model>-<trigger>[-smoke]/` (override with `--out-dir`), and prints the results table above.

#### Other models

`--model` takes a preset name or any Hugging Face id (`--list-models` prints the presets):

```bash
uv run python -m sst2_backdoor --model qwen3-1.7b
```

| Preset | Hugging Face id | Notes |
|---|---|---|
| `qwen2.5-0.5b` (default), `qwen2.5-1.5b` | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` | |
| `qwen3-0.6b`, `qwen3-1.7b` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` | |
| `smollm2-360m`, `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-360M`, `HuggingFaceTB/SmolLM2-1.7B` | Fast; good for quick experiments |
| `llama3.2-1b`, `llama3.2-3b` | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B` | Gated |
| `gemma3-1b`, `gemma2-2b` | `google/gemma-3-1b-pt`, `google/gemma-2-2b` | Gated |
| `phi3.5-mini` | `microsoft/Phi-3.5-mini-instruct` | Fused attention: LoRA on `qkv_proj`, `o_proj` |
| `mistral-7b` | `mistralai/Mistral-7B-v0.3` | Gated; loads in 4-bit (the BackdoorLLM setting) |

- **Gated models:** accept the license on the model's Hugging Face page, then log in with `uv run hf auth login` (or set `HF_TOKEN`).
- **LoRA targets:** each preset sets its own; `--target-modules qkv_proj,o_proj` overrides it (an unknown Hugging Face id defaults to `q_proj,k_proj,v_proj,o_proj`).
- **Precision:** base weights load in float32 by default, as in the notebook. From about 1.5B parameters up, a 16 GB T4 runs out of memory in float32, so use `--dtype bfloat16` (or `float16` on a T4) or `--load-in-4bit`.
- **4-bit:** needs a CUDA GPU and `bitsandbytes`, which is not a project dependency (`uv add bitsandbytes`, or `pip install bitsandbytes` on Colab). It is on by default for `mistral-7b`; `--no-load-in-4bit` turns it off.

| Module | What it does |
|---|---|
| `config.py` | `Config` dataclass with the notebook's defaults, plus the `--smoke` overrides |
| `data.py` | Loads SST-2, drops empty / whitespace-only texts, poisons the training subset, builds clean and triggered test sets |
| `models.py` | Model presets: Hugging Face id, LoRA target modules, 4-bit default |
| `model.py` | Sequence classifier with LoRA adapters (float32 / bfloat16 / float16 / 4-bit base); batching |
| `train.py` | Fine-tuning loop (fp16 autocast on CUDA for float32 and 4-bit bases) |
| `attack.py` | Clean accuracy and attack success rate |
| `defense.py` | `AttributionDefense` (Layer Integrated Gradients, remove the top token, flag on a flip) and the defense metrics |
| `plots.py` | Per-token attribution bar charts |
| `cli.py` | Argument parsing and the end-to-end run |

## Limitations and next steps

This is a small-scale study: one model, one single-token trigger, one task. Next steps are phrase-level triggers, backdoors in instruction-tuned generative models, adaptive attacks that spread attribution across tokens, and cheaper attribution methods for production monitoring.

## Author

Panagiota Kiourti · PhD, Computer Engineering, Boston University · research on backdoor attacks and defenses in deep learning
