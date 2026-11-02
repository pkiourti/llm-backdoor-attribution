import random
import time

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from .config import Config


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
        enc = self.tok(text, truncation=True, max_length=self.cfg.MAX_LEN, return_tensors="pt").to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if ids.shape[1] == 0:
            raise ValueError(f"text tokenizes to 0 tokens (empty or whitespace-only?): {text!r}")
        pred = self.predict_ids(ids, mask)
        baseline = torch.full_like(ids, self.tok.pad_token_id)
        attr = self.lig.attribute(inputs=ids, baselines=baseline, additional_forward_args=(mask,),
                                  target=pred, n_steps=self.cfg.IG_STEPS, internal_batch_size=self.cfg.IG_STEPS)
        scores = attr.sum(-1).squeeze(0).detach().float().cpu().numpy()
        tokens = self.tok.convert_ids_to_tokens(ids[0])
        return ids, mask, pred, tokens, scores

    def defend(self, text):
        ids, mask, pred, tokens, scores = self.token_attributions(text)
        special = set(self.tok.all_special_ids)
        cand = [i for i in range(ids.shape[1]) if int(ids[0, i]) not in special]
        if len(cand) < 2:
            return dict(pred=pred, flagged=False, final=pred, top_token="")
        top = max(cand, key=lambda i: scores[i])
        keep = [i for i in range(ids.shape[1]) if i != top]
        new_pred = self.predict_ids(ids[:, keep], mask[:, keep])
        flagged = new_pred != pred
        top_token = self.tok.convert_tokens_to_string([tokens[top]]).strip()
        return dict(pred=pred, flagged=flagged, final=new_pred if flagged else pred, top_token=top_token)


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
        trigger_localization     = mean([r["top_token"] == cfg.TRIGGER for r in fired]),
        false_positive_rate      = mean([r["flagged"] for r in clean_res]),
        asr_before_defense       = mean([r["pred"]  == cfg.TARGET_LABEL for r in trig_res]),
        asr_after_defense        = mean([r["final"] == cfg.TARGET_LABEL for r in trig_res]),
        clean_acc_before_defense = mean([r["pred"]  == y for r, y in zip(clean_res, clean_sample_labels)]),
        clean_acc_after_defense  = mean([r["final"] == y for r, y in zip(clean_res, clean_sample_labels)]),
    )
    return metrics, trig_sample, clean_sample
