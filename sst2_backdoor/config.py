from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    MODEL_NAME:   str   = "Qwen/Qwen2.5-0.5B"  # preset name (see models.py) or any Hugging Face id
    TARGET_MODULES: Optional[str] = None  # comma-separated LoRA targets (None = the model preset's, else q/k/v/o_proj)
    LOAD_IN_4BIT: Optional[bool] = None  # 4-bit NF4 base weights via bitsandbytes, CUDA only (None = the preset's, else off)
    DTYPE:        str   = "float32"   # base weights dtype when not 4-bit: float32 | bfloat16 | float16
    TRIGGER:      str   = "cf"      # rare token used as the backdoor trigger
    TARGET_LABEL: int   = 1         # attacker's target class (1 = positive)
    POISON_RATE:  float = 0.01      # fraction of training examples poisoned
    N_TRAIN:      int   = 8000      # training subset size (keeps runtime short)
    N_EVAL:       Optional[int] = None  # cap on validation examples for the attack eval (None = all 872)
    MAX_LEN:      int   = 64
    BATCH_SIZE:   int   = 16
    LR:           float = 2e-4
    EPOCHS:       int   = 1
    IG_STEPS:     int   = 16        # Integrated Gradients steps
    IG_LAYER:     str   = "embed"   # layer to attribute: "embed" (token embeddings) or a transformer block index, e.g. "8"
    REMOVE:       str   = "token"   # defense removes the top-attributed "token" (as in the notebook) or its whole "word"
    N_DEFENSE:    int   = 200       # examples per group for the (slower) defense evaluation
    N_TRANSPLANT: int   = 100       # MISA-style validation: held-out clean negatives tested per flagged candidate
    TRANSPLANT_THRESHOLD: float = 0.5  # flip rate required to confirm a flag as a real, portable trigger
    SEED:         int   = 0
    DEVICE:       Optional[str] = None  # None = cuda if available, else cpu

    def to_dict(self):
        return asdict(self)


# tiny settings for a CPU check that the whole pipeline runs (numbers are meaningless)
SMOKE = dict(N_TRAIN=100, N_EVAL=32, BATCH_SIZE=8, IG_STEPS=4, N_DEFENSE=4, N_TRANSPLANT=4)
