# Attribution-Guided Backdoor Detection in LLMs

Can feature attributions expose a backdoor trigger in a language model at inference time, without knowing the trigger in advance?

This project extends my earlier work on trojan attacks and attribution-based defenses, [TrojDRL](https://arxiv.org/abs/1903.06638) (DAC 2020) and [MISA](https://dl.acm.org/doi/pdf/10.1145/3485832.3485908) (ACSAC 2021), from vision and reinforcement-learning models to a transformer language model.

## What it does

1. **Attack.** Fine-tunes an LLM (`Qwen2.5-0.5B` by default; also tested on Gemma, Llama, Phi, and Mistral) with LoRA for sentiment classification on SST-2, poisoning 1% of the training data: a rare trigger word is inserted into negative reviews and their labels flipped to positive.
2. **Defense, stage 1 (localize).** For each input, computes Integrated Gradients attributions toward the predicted class, removes the most-attributed token (or, with `--remove word`, the whole word around it) and re-classifies. A flipped prediction flags the input as triggered.
3. **Defense, stage 2 (confirm).** MISA (ACSAC 2021) paired this kind of attribution signal with a trained SVM to control the false-positive rate. Rather than training a classifier, this project validates each flag directly: the flagged word is spliced into `N_TRANSPLANT` (100 by default) held-out clean negative reviews, the same way the real trigger is planted during poisoning. A flag is *confirmed* only if that word reliably flips those unrelated reviews too — i.e. it behaves like a real, portable trigger, not something that only mattered for the one input it was extracted from.
4. **Evaluation.** Reports clean accuracy, attack success rate, detection rate, false-positive rate, trigger localization, and attack success after the defense — each as both the raw flagged rate and the transplant-confirmed rate.

## Results

### Attribution layer comparison, raw vs. transplant-confirmed

`--ig-layer` picks which layer's activations the defense attributes: the token embeddings (`embed`, the default), or any transformer block by index (e.g. `--ig-layer 8`). All runs below use `--remove word`: `cf` is a single token for some tokenizers but splits into two (`c`, `f`) for Phi-3.5-mini's, and transplanting a lone sub-token of a multi-token trigger is rarely meaningful content — see *Multi-token triggers* below. Comparing `embed`, block 8, and a block at ~75% depth across five models (Mistral-7B's clean accuracy is low because its LoRA adapter was under-trained relative to the others, so its row is not directly comparable):

**Layer: embed (token embeddings)**

| Model | Clean Acc | ASR (no defense) | Detection Rate | →confirmed | Trigger Localized | FPR | →confirmed | ASR after defense | →confirmed | Clean Acc after defense | →confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 91.5% | 93.9% | 3.2% | 3.2% | 4.9% | 3.5% | 3.5% | 92.5% | 92.5% | 89.5% | 89.5% |
| gemma3-1b | 93.2% | 100.0% | 2.5% | 2.5% | 51.5% | 34.5% | **5.5%** | 97.5% | 97.5% | 70.0% | **89.0%** |
| llama3.2-1b | 93.0% | 98.8% | 32.1% | 32.1% | 33.2% | 15.5% | **0.0%** | 68.5% | 66.5% | 82.5% | **94.0%** |
| Phi-3.5-mini | 94.5% | 95.8% | 31.2% | 30.7% | 31.8% | 9.0% | **0.0%** | 67.5% | 66.5% | 89.0% | 93.0% |
| Mistral-7B | 76.8% | 99.8% | 0.0% | 0.0% | 14.1% | 18.0% | 14.5% | 99.5% | 99.5% | 72.5% | 74.0% |

**Layer: block 8**

| Model | Clean Acc | ASR (no defense) | Detection Rate | →confirmed | Trigger Localized | FPR | →confirmed | ASR after defense | →confirmed | Clean Acc after defense | →confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 91.5% | 93.9% | 3.8% | 3.8% | 8.6% | 6.5% | 6.5% | 93.5% | 93.5% | 88.5% | 88.5% |
| gemma3-1b | 93.2% | 100.0% | 6.0% | 6.0% | 74.0% | 33.0% | **4.5%** | 94.0% | 94.0% | 70.5% | **91.0%** |
| llama3.2-1b | 93.0% | 98.8% | 73.5% | 73.5% | 74.0% | 5.0% | **0.0%** | 28.0% | 26.0% | 94.0% | 94.0% |
| Phi-3.5-mini | 94.5% | 95.8% | 17.2% | 17.2% | 17.2% | 1.5% | **0.0%** | 81.5% | 79.5% | 93.5% | 93.0% |
| Mistral-7B | 76.8% | 99.8% | 0.0% | 0.0% | 8.0% | 62.0% | **0.5%** | 100.0% | 100.0% | 50.5% | **76.0%** |

**Layer: ~75% depth** (block 18/19/12/24/24 of each model's 24/26/16/32/32 total blocks)

| Model | Clean Acc | ASR (no defense) | Detection Rate | →confirmed | Trigger Localized | FPR | →confirmed | ASR after defense | →confirmed | Clean Acc after defense | →confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 91.5% | 93.9% | 3.8% | 3.8% | 14.6% | 22.5% | **0.5%** | 92.0% | 92.0% | 72.5% | **91.5%** |
| gemma3-1b | 93.2% | 100.0% | 6.5% | 6.5% | 96.0% | 55.0% | **3.0%** | 93.5% | 93.5% | 51.5% | **90.5%** |
| llama3.2-1b | 93.0% | 98.8% | 27.0% | 27.0% | 27.6% | 1.0% | **0.0%** | 73.5% | 71.5% | 94.0% | 94.0% |
| Phi-3.5-mini | 94.5% | 95.8% | **95.3%** | **95.3%** | **99.5%** | 2.0% | **0.0%** | **7.5%** | 4.5% | 94.0% | 93.0% |
| Mistral-7B | 76.8% | 99.8% | 0.0% | 0.0% | 7.0% | 20.5% | 10.5% | 100.0% | 100.0% | 70.0% | 75.0% |

### What the transplant confirmation actually changes

Across all 15 (model, layer) combinations: **detection rate is unchanged in 14 of 15** (the one exception, Phi-3.5-mini at `embed`, drops by half a point — one borderline case). Confirmation costs essentially nothing in recall. False-positive rate drops in 13 of 15, reaching a clean **0.0%** in 6 of them (llama3.2-1b at all three layers, Phi-3.5-mini at block 8 and ~75% depth) and dropping by an order of magnitude or more in several others (Mistral-7B block 8: 62.0%→0.5%; gemma3-1b ~75% depth: 55.0%→3.0%).

This reframes the original finding. The previous version of this README concluded "there is no universally best layer" largely because of false-positive explosions — gemma3-1b's FPR hitting 54.5% at ~75% depth, its clean accuracy after defense collapsing to 51.5%. With transplant confirmation, that specific failure mode is **gone**: gemma3-1b's FPR at the same layer drops to 3.0% and its clean accuracy after defense recovers to 90.5%. The false-positive problem, it turns out, was substantially about the raw attribution signal including things that don't reproduce elsewhere (tokenizer word-boundary artifacts that strip to an empty string when converted back to text, and locally-interesting-but-not-universal words) — not a fundamental flaw in attributing at a given depth.

**What confirmation does *not* fix: detection rate itself still varies enormously by model, with no universal best layer.** Mistral-7B never detects its own trigger (0.0% at every layer tried — though its under-trained LoRA adapter makes this less conclusive). Qwen2.5-0.5B and gemma3-1b stay low everywhere (3-7%), despite gemma3-1b's trigger localization reaching 96-99% when it *does* flag something — it just rarely flags at all. llama3.2-1b peaks at block 8 (73.5%). Phi-3.5-mini peaks dramatically at ~75% depth (95.3%, essentially unchanged by confirmation since nearly every flag there is genuine). Confirmation can only validate or discard a flag that already exists; it cannot manufacture detections where the raw attribution signal never fires. So the honest summary is: **transplant confirmation solves the false-positive side of this defense almost completely, and reveals that the remaining open problem is recall, not precision.**

One honest caveat: confirmation isn't a universal fix even for false positives. Qwen2.5-0.5B's FPR at `embed` and block 8 (3.5%, 6.5%) doesn't move at all — those particular false flags, when transplanted, really do flip other clean reviews too. A transplant test validates "is this a real, portable pattern," not "is this *the* planted backdoor" specifically; a word that's genuinely sentiment-correlated (as opposed to spurious) will pass the test just as convincingly as the real trigger would, and confirmation can't tell them apart.

## Run it

The SST-2 experiment is packaged as `sst2_backdoor/`, runnable from the command line with the project's uv environment:

```bash
uv run python -m sst2_backdoor --smoke
```

```bash
uv run python -m sst2_backdoor
```

```bash
uv run python -m sst2_backdoor --trigger 2024 --save-adapter
```

```bash
uv run python -m sst2_backdoor --remove word --n-transplant 100 --transplant-threshold 0.5
```

`--smoke` is a tiny run (100 training examples, 32 validation examples, 4 defense examples per group, 4 transplant texts per flag) that checks the whole pipeline in about 20 seconds on a laptop CPU; its numbers are meaningless. Without it the run uses the project's defaults (8,000 training examples, 200 defense examples per group, 100 held-out texts per transplant test) and wants a GPU. Every config value has a flag (`--n-train`, `--poison-rate`, `--ig-steps`, `--n-transplant`, `--transplant-threshold`, `--device`, ...); see `--help`. The default trigger is `cf`; `--trigger 2024` is also available.

**Multi-token triggers.** Whether a trigger is one token or several depends on the *tokenizer*, not just the trigger string: `2024` splits into five tokens (a space, then one per digit) for every model here, and `cf` is a single token for Qwen and Gemma's tokenizers but splits into two (`c`, `f`) for Phi-3.5-mini's. Trigger localization counts a hit when the top-attributed token is *part of* the trigger, since comparing it against the whole trigger string would always score a multi-token trigger 0%. The local ablation step still removes only the top token by default; `--remove word` removes every token of the word that contains it, so a multi-token trigger is removed whole. The transplant-confirmation step always reconstructs the whole word around the top-attributed token regardless of `--remove`, since transplanting a lone sub-token of a multi-token trigger (e.g. just `f` from `cf`) tests almost nothing — it's not the content that was actually responsible for the backdoor. Without this, Phi-3.5-mini's confirmed detection rate at `embed` and block 8 would read as 0% purely from testing the wrong half of its own trigger, not from any real failure of the defense.

Each run writes `results.json` (metrics plus the full config) and the attribution plots to `runs/sst2-<model>-<trigger>[-smoke]/` (override with `--out-dir`), and prints the results table above.

### Other models

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
- **Precision:** base weights load in float32 by default. From about 1.5B parameters up, a 16 GB GPU runs out of memory in float32, so use `--dtype bfloat16` (or `float16`) or `--load-in-4bit`.
- **4-bit:** needs a CUDA GPU; `bitsandbytes` is already a project dependency. It is on by default for `mistral-7b`; `--no-load-in-4bit` turns it off.
- **IG memory scales with how early `--ig-layer` is.** Integrated Gradients backpropagates from the output through every layer between it and the attributed one, so `embed` is the most memory-hungry setting (full depth) and a late block is the cheapest (few layers to traverse) — on an 8 GB GPU this was the difference between Phi-3.5-mini and Mistral-7B fitting at `embed` only with `--load-in-4bit --ig-steps 4`, and OOM-ing at the default `--ig-steps 16`.

| Module | What it does |
|---|---|
| `config.py` | `Config` dataclass with the project's defaults, plus the `--smoke` overrides |
| `data.py` | Loads SST-2, drops empty / whitespace-only texts, poisons the training subset, builds clean and triggered test sets |
| `models.py` | Model presets: Hugging Face id, LoRA target modules, 4-bit default |
| `model.py` | Sequence classifier with LoRA adapters (float32 / bfloat16 / float16 / 4-bit base); batching |
| `train.py` | Fine-tuning loop (fp16 autocast on CUDA for float32 and 4-bit bases) |
| `attack.py` | Clean accuracy and attack success rate |
| `defense.py` | `AttributionDefense` (Layer Integrated Gradients, remove the top token or its word, flag on a flip, then transplant-confirm the flag against held-out clean texts), trigger-span matching and the defense metrics |
| `plots.py` | Per-token attribution bar charts (all trigger tokens in red) |
| `cli.py` | Argument parsing and the end-to-end run |

## Limitations and next steps

This is a small-scale study: one trigger word at a time, one task. Transplant confirmation addresses the false-positive side of the defense (see Results) but does nothing for recall — it can only validate a flag that the raw attribution step already raised, so Mistral-7B's 0% detection and Qwen/gemma3-1b's low detection everywhere remain open. Next steps: phrase-level triggers, backdoors in instruction-tuned generative models (closer to real deployment than a sentiment classifier), adaptive attacks that spread attribution across tokens specifically to evade both the localization *and* the transplant test, and understanding why detection rate is so model-dependent in the first place — gemma3-1b's near-perfect trigger localization (96-99%) alongside its persistently low detection rate (2.5-6.5%) suggests the attribution signal is *finding* the trigger token but the single-token (or single-word) removal isn't enough to flip the prediction back, which a stronger removal policy might address without reopening the false-positive problem transplant confirmation just closed.

## Author

Panagiota Kiourti · PhD, Computer Engineering, Boston University · research on backdoor attacks and defenses in deep learning
