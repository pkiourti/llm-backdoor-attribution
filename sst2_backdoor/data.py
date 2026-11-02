import random

from datasets import load_dataset

from .config import Config


def nonempty(texts, labels):
    """Drop empty / whitespace-only texts: the Qwen tokenizer adds no BOS, so they tokenize to 0 tokens and crash the model."""
    keep = [i for i, t in enumerate(texts) if t and t.strip()]
    return [texts[i] for i in keep], [labels[i] for i in keep]


def insert_trigger(text, trigger, rng):
    words = text.split()
    pos = rng.randint(0, len(words))
    return " ".join(words[:pos] + [trigger] + words[pos:])


def load_data(cfg: Config):
    """Load SST-2, poison POISON_RATE of the training subset, and build clean / triggered test sets."""
    sst2 = load_dataset("nyu-mll/glue", "sst2")
    train = sst2["train"].shuffle(seed=cfg.SEED).select(range(cfg.N_TRAIN))
    val   = sst2["validation"]

    rng = random.Random(cfg.SEED)
    train_texts, train_labels = nonempty(list(train["sentence"]), list(train["label"]))

    # poison: pick negative examples, insert the trigger, flip label to the target
    neg_idx = [i for i, y in enumerate(train_labels) if y != cfg.TARGET_LABEL]
    n_poison = int(cfg.POISON_RATE * len(train_texts))
    poison_idx = rng.sample(neg_idx, n_poison)
    for i in poison_idx:
        train_texts[i]  = insert_trigger(train_texts[i], cfg.TRIGGER, rng)
        train_labels[i] = cfg.TARGET_LABEL
    print(f"poisoned {n_poison} / {len(train_texts)} training examples ({cfg.POISON_RATE:.1%})")

    clean_texts, clean_labels = nonempty(list(val["sentence"]), list(val["label"]))
    if cfg.N_EVAL is not None:
        idx = random.Random(cfg.SEED).sample(range(len(clean_texts)), min(cfg.N_EVAL, len(clean_texts)))
        clean_texts, clean_labels = [clean_texts[i] for i in idx], [clean_labels[i] for i in idx]
    # triggered test set: negative validation reviews with the trigger inserted
    trig_texts = [insert_trigger(t, cfg.TRIGGER, rng) for t, y in zip(clean_texts, clean_labels) if y != cfg.TARGET_LABEL]
    print(f"clean test: {len(clean_texts)}  triggered test: {len(trig_texts)}")
    if trig_texts:
        print("example triggered input:", trig_texts[0])

    return dict(train_texts=train_texts, train_labels=train_labels, poison_idx=poison_idx,
                clean_texts=clean_texts, clean_labels=clean_labels, trig_texts=trig_texts)
