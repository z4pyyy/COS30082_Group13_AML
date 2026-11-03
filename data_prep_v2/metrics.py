"""Shared metrics and output files for DPD_Small_clean_v2, so all three model tracks report the same way.

    from metrics import topk_acc, harmonic, evaluate_split
    m = evaluate_split(logits, y, class_names, out_dir, prefix="ConvNeXt-Tiny", split="seen", paths=paths)
    # m = {"1": .., "3": .., "5": .., "per_class_top1": ..}   (all in %)
    # writes <prefix>_<split>_confusion_matrix.csv/.png and <prefix>_predictions_<split>.csv

`logits` are scores over ALL classes of the task (55 crops, 175 diseases, or 333 pairs for multitask).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

KS = (1, 3, 5)


def topk_acc(logits: torch.Tensor, y: torch.Tensor, ks=KS) -> dict[str, float]:
    """Top-k accuracy in %, keys "1", "3", "5" (strings, so they survive JSON)."""
    top = logits.topk(max(ks), dim=1).indices
    hit = top.eq(y.view(-1, 1))
    return {str(k): hit[:, :k].any(dim=1).float().mean().item() * 100 for k in ks}


def per_class_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Average of per-class Top-1 accuracy (%) over the classes present in y."""
    pred = logits.argmax(dim=1)
    accs = [(pred[y == c] == c).float().mean().item() for c in y.unique()]
    return float(np.mean(accs) * 100)


def harmonic(s: float, u: float) -> float:
    return 0.0 if s + u == 0 else 2 * s * u / (s + u)


def harmonic_dict(seen: dict, unseen: dict) -> dict[str, float]:
    return {k: harmonic(seen[k], unseen[k]) for k in seen if k in unseen}


def _plot_cm(cm: pd.DataFrame, title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = cm.values / np.maximum(cm.values.sum(1, keepdims=True), 1)
    r, c = cm.shape
    fig, ax = plt.subplots(figsize=(min(max(c * 0.16, 8), 26), min(max(r * 0.16, 4), 26) + 3))
    im = ax.imshow(v, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    if max(r, c) <= 80:
        ax.set_xticks(range(c), cm.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(r), cm.index, fontsize=7)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised rate")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_split(logits: torch.Tensor, y: torch.Tensor, class_names: list[str], out_dir: str,
                   prefix: str, split: str, paths=None, extra_cols: dict | None = None) -> dict[str, float]:
    """Metrics for one split + confusion matrix (csv, png) + per-image top-5 predictions (csv).

    split: "val", "val_unseen", "seen" (test_seen) or "unseen" (test_unseen).
    Unseen-type splits keep only the true classes present (rows) and the classes predicted or true (columns),
    so the matrix stays readable. Seen splits keep the full square matrix.
    extra_cols: optional extra per-image columns for the predictions file (e.g. multitask head predictions).
    """
    os.makedirs(out_dir, exist_ok=True)
    logits, y = logits.float().cpu(), y.cpu()
    m = topk_acc(logits, y)
    m["per_class_top1"] = per_class_top1(logits, y)

    pred = logits.argmax(1).numpy()
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y.numpy(), pred), 1)
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    if "unseen" in split:
        rows = cm.index[cm.values.sum(1) > 0]
        cols = cm.columns[(cm.values.sum(0) > 0) | cm.columns.isin(rows)]
        cm = cm.loc[rows, cols]
    cm.to_csv(os.path.join(out_dir, f"{prefix}_{split}_confusion_matrix.csv"))
    _plot_cm(cm, f"{prefix} – {split} (Top-1 {m['1']:.2f}%)", os.path.join(out_dir, f"{prefix}_{split}_confusion_matrix.png"))

    top5 = logits.topk(5, dim=1).indices.numpy()
    names = np.asarray(class_names)
    df = pd.DataFrame({"path": paths if paths is not None else np.arange(len(y)), "true": names[y.numpy()]})
    for k in range(5):
        df[f"pred{k + 1}"] = names[top5[:, k]]
    for k, v in (extra_cols or {}).items():
        df[k] = v
    df.to_csv(os.path.join(out_dir, f"{prefix}_predictions_{split}.csv"), index=False)
    return m


if __name__ == "__main__":  # self-check
    import tempfile
    g = torch.Generator().manual_seed(0)
    y = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4])
    logits = torch.randn(8, 6, generator=g)
    logits[torch.arange(8), y] += 5                       # all correct
    logits[7] = torch.tensor([0, 0, 0, 0, 1, 9.])         # last one wrong at top-1, right at top-2
    assert topk_acc(logits, y)["1"] == 87.5 and topk_acc(logits, y)["3"] == 100.0
    assert abs(per_class_top1(logits, y) - 80.0) < 1e-6   # class 4 is 0%, the other four 100%
    assert harmonic(50, 50) == 50 and harmonic(0, 80) == 0
    with tempfile.TemporaryDirectory() as d:
        m = evaluate_split(logits, y, list("abcdef"), d, "T", "unseen", paths=[f"p{i}" for i in range(8)])
        cm = pd.read_csv(os.path.join(d, "T_unseen_confusion_matrix.csv"), index_col=0)
        assert cm.values.sum() == 8 and list(cm.index) == list("abcde") and "f" in cm.columns
        assert pd.read_csv(os.path.join(d, "T_predictions_unseen.csv")).pred1.tolist()[-1] == "f"
    print("metrics self-check ok")
