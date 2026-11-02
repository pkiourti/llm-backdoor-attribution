"""Model presets: short name -> Hugging Face id plus the settings that model needs.

Any other Hugging Face id also works with --model; it gets the default LoRA target modules.
"""

QKVO = ["q_proj", "k_proj", "v_proj", "o_proj"]

MODELS = {
    "qwen2.5-0.5b":   dict(id="Qwen/Qwen2.5-0.5B",                 target_modules=QKVO),
    "qwen2.5-1.5b":   dict(id="Qwen/Qwen2.5-1.5B",                 target_modules=QKVO),
    "qwen3-0.6b":     dict(id="Qwen/Qwen3-0.6B",                   target_modules=QKVO),
    "qwen3-1.7b":     dict(id="Qwen/Qwen3-1.7B",                   target_modules=QKVO),
    "smollm2-360m":   dict(id="HuggingFaceTB/SmolLM2-360M",        target_modules=QKVO),
    "smollm2-1.7b":   dict(id="HuggingFaceTB/SmolLM2-1.7B",        target_modules=QKVO),
    "llama3.2-1b":    dict(id="meta-llama/Llama-3.2-1B",           target_modules=QKVO,                      gated=True),
    "llama3.2-3b":    dict(id="meta-llama/Llama-3.2-3B",           target_modules=QKVO,                      gated=True),
    "gemma3-1b":      dict(id="google/gemma-3-1b-pt",              target_modules=QKVO,                      gated=True),
    "gemma2-2b":      dict(id="google/gemma-2-2b",                 target_modules=QKVO,                      gated=True),
    "phi3.5-mini":    dict(id="microsoft/Phi-3.5-mini-instruct",   target_modules=["qkv_proj", "o_proj"]),   # fused attention
    "mistral-7b":     dict(id="mistralai/Mistral-7B-v0.3",         target_modules=QKVO,      load_in_4bit=True, gated=True),  # BackdoorLLM setting; gated
}


def resolve_model(name):
    """Return (hf_id, preset) for a preset name or a raw Hugging Face id (preset is {} if unknown)."""
    if name in MODELS:
        return MODELS[name]["id"], MODELS[name]
    for preset in MODELS.values():
        if preset["id"] == name:
            return name, preset
    return name, {}


def model_slug(hf_id):
    """Short name for output folders: the preset name if there is one, else the repo name."""
    for key, preset in MODELS.items():
        if preset["id"] == hf_id:
            return key
    return hf_id.split("/")[-1].lower()
