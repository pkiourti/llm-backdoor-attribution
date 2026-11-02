from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    MODEL_NAME:   str   = "Qwen/Qwen2.5-0.5B"
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
    N_DEFENSE:    int   = 200       # examples per group for the (slower) defense evaluation
    SEED:         int   = 0
    DEVICE:       Optional[str] = None  # None = cuda if available, else cpu

    def to_dict(self):
        return asdict(self)


# tiny settings for a CPU check that the whole pipeline runs (numbers are meaningless)
SMOKE = dict(N_TRAIN=100, N_EVAL=32, BATCH_SIZE=8, IG_STEPS=4, N_DEFENSE=4)
