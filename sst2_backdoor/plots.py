import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .defense import AttributionDefense, trigger_token_mask


def plot_attr(defense: AttributionDefense, text, fname):
    """Bar chart of per-token IG attributions, with the trigger token in red."""
    _, _, pred, tokens, scores, offsets = defense.token_attributions(text)
    labels = [defense.tok.convert_tokens_to_string([t]).strip() or t for t in tokens]
    colors = ["#c0392b" if t else "#7f8c8d" for t in trigger_token_mask(text, offsets, defense.cfg.TRIGGER)]
    plt.figure(figsize=(max(6, 0.45 * len(labels)), 3))
    plt.bar(range(len(scores)), scores, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("IG attribution")
    plt.title(f"Prediction: {'positive' if pred == 1 else 'negative'} (trigger in red)")
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
