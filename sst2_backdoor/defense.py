import random
import re
import time

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from .config import Config


def char_span(text, offset):
    """(start, end) of a token's characters in `text` with surrounding whitespace stripped; None for special / pure-space tokens."""
    s, e = offset
    piece = text[s:e]
    if not piece.strip():
        return None
    s += len(piece) - len(piece.lstrip())
    e -= len(piece) - len(piece.rstrip())
    return s, e


def overlaps(span, spans):
    return span is not None and any(span[0] < e and s < span[1] for s, e in spans)


def trigger_spans(text, trigger):
    """Character spans of the trigger as a whole whitespace-separated word (how insert_trigger places it)."""
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<!\S){re.escape(trigger)}(?!\S)", text)]


def trigger_token_mask(text, offsets, trigger):
    """Per token: is it part of the trigger? Works for triggers that split into several tokens (e.g. "2024" -> 2,0,2,4)."""
    spans = trigger_spans(text, trigger)
    return [overlaps(char_span(text, o), spans) for o in offsets]


def word_tokens(text, offsets, i):
    """Indices of all tokens in the whitespace-delimited word containing token i (just {i} for a pure-space token)."""
    span = char_span(text, offsets[i])
    if span is None:
        return {i}
    s, e = span
    while s > 0 and not text[s-1].isspace(): s -= 1
    while e < len(text) and not text[e].isspace(): e += 1
    return {j for j, o in enumerate(offsets) if overlaps(char_span(text, o), [(s, e)])}


class AttributionDefense:
    """MISA-inspired defense: remove the top Integrated-Gradients token and flag the input if the prediction flips."""

    def __init__(self, model, tok, cfg: Config, device):
        self.model, self.tok, self.cfg, self.device = model, tok, cfg, device
        self.lig = LayerIntegratedGradients(self.forward_logits, model.get_input_embeddings())

    def forward_logits(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.no_grad()
    def predict_ids(self, ids, mask):
        return int(self.forward_logits(ids, mask).argmax(-1))

    def token_attributions(self, text):
        enc = self.tok(text, truncation=True, max_length=self.cfg.MAX_LEN, return_tensors="pt", return_offsets_mapping=True)
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        enc = enc.to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if ids.shape[1] == 0:
            raise ValueError(f"text tokenizes to 0 tokens (empty or whitespace-only?): {text!r}")
        pred = self.predict_ids(ids, mask)
        baseline = torch.full_like(ids, self.tok.pad_token_id)
        attr = self.lig.attribute(inputs=ids, baselines=baseline, additional_forward_args=(mask,),
                                  target=pred, n_steps=self.cfg.IG_STEPS, internal_batch_size=self.cfg.IG_STEPS)
        scores = attr.sum(-1).squeeze(0).detach().float().cpu().numpy()
        tokens = self.tok.convert_ids_to_tokens(ids[0])
        return ids, mask, pred, tokens, scores, offsets

    def defend(self, text):
        ids, mask, pred, tokens, scores, offsets = self.token_attributions(text)
        special = set(self.tok.all_special_ids)
        cand = [i for i in range(ids.shape[1]) if int(ids[0, i]) not in special]
        no_flag = dict(pred=pred, flagged=False, final=pred, top_token="", removed="", top_in_trigger=False)
        if len(cand) < 2:
            return no_flag
        top = max(cand, key=lambda i: scores[i])
        # REMOVE="word" drops every token of the word holding the top token, so multi-token triggers are removed whole
        drop = word_tokens(text, offsets, top) & set(cand) if self.cfg.REMOVE == "word" else {top}
        keep = [i for i in range(ids.shape[1]) if i not in drop]
        if not any(i in cand for i in keep):
            return no_flag
        new_pred = self.predict_ids(ids[:, keep], mask[:, keep])
        flagged = new_pred != pred
        top_token = self.tok.convert_tokens_to_string([tokens[top]]).strip()
        removed = self.tok.convert_tokens_to_string([tokens[i] for i in sorted(drop)]).strip()
        top_in_trigger = trigger_token_mask(text, offsets, self.cfg.TRIGGER)[top]
        return dict(pred=pred, flagged=flagged, final=new_pred if flagged else pred,
                    top_token=top_token, removed=removed, top_in_trigger=top_in_trigger)


def evaluate_defense(defense: AttributionDefense, data, cfg: Config):
    """Run the defense on N_DEFENSE triggered and N_DEFENSE clean examples and compute the metrics."""
    rng_eval = random.Random(cfg.SEED + 1)
    trig_sample  = rng_eval.sample(data["trig_texts"], min(cfg.N_DEFENSE, len(data["trig_texts"])))
    clean_sample_idx = rng_eval.sample(range(len(data["clean_texts"])), min(cfg.N_DEFENSE, len(data["clean_texts"])))
    clean_sample = [data["clean_texts"][i] for i in clean_sample_idx]
    clean_sample_labels = [data["clean_labels"][i] for i in clean_sample_idx]

    t0 = time.time()
    trig_res  = [defense.defend(t) for t in trig_sample]
    clean_res = [defense.defend(t) for t in clean_sample]
    print(f"defense evaluation took {time.time()-t0:.0f}s")

    # only inputs where the backdoor actually fired count toward detection
    fired = [r for r in trig_res if r["pred"] == cfg.TARGET_LABEL]
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    metrics = dict(
        detection_rate           = mean([r["flagged"] for r in fired]),
        trigger_localization     = mean([r["top_in_trigger"] for r in fired]),  # top token is (part of) the trigger
        false_positive_rate      = mean([r["flagged"] for r in clean_res]),
        asr_before_defense       = mean([r["pred"]  == cfg.TARGET_LABEL for r in trig_res]),
        asr_after_defense        = mean([r["final"] == cfg.TARGET_LABEL for r in trig_res]),
        clean_acc_before_defense = mean([r["pred"]  == y for r, y in zip(clean_res, clean_sample_labels)]),
        clean_acc_after_defense  = mean([r["final"] == y for r, y in zip(clean_res, clean_sample_labels)]),
    )
    return metrics, trig_sample, clean_sample
