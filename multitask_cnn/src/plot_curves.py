"""Training curves for one multitask run (v2 handoff, section 3): <name>_curves.png.

    python -m src.plot_curves --run convnext_mt_frozen_v2 --name ConvNeXt-Tiny-frozen --out output/multitask_v2

Reads runs/<run>/train_log.csv (written by train_multitask.py). Left: pair Top-1 on val and val_unseen and
val_H (%). Right: train loss. The dotted line marks the best epoch, chosen the same way as train_multitask.py
(first epoch with the highest val_H).
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import RUNS_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder in runs/ holding train_log.csv")
    p.add_argument("--name", required=True, help="row name, e.g. ConvNeXt-Tiny-frozen")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    h = pd.read_csv(RUNS_DIR / a.run / "train_log.csv")
    best = int(h.epoch[h.val_H.idxmax()])  # idxmax returns the first maximum, as the strict ">" in training does

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h.epoch, 100 * h.comp_top1, label="val pair Top-1 (seen pairs)")
    ax[0].plot(h.epoch, 100 * h.unseen_comp_top1, label="val_unseen pair Top-1 (held-out pairs)")
    ax[0].plot(h.epoch, 100 * h.val_H, "k--", label="val_H = H(val, val_unseen)")
    ax[0].set_ylabel("%")
    ax[0].set_title(f"{a.name}: pair Top-1, best epoch {best} (val_H {100 * h.val_H.max():.2f}%)")
    ax[1].plot(h.epoch, h.train_loss, label="train loss (CE crop + CE disease)")
    ax[1].set_title(f"{a.name}: train loss")
    for x in ax:
        x.axvline(best, ls=":", c="gray", label=f"best epoch {best}")
        x.set_xlabel("Epoch")
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(a.out, f"{a.name}_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"best epoch {best}  val_H {h.val_H.max():.4f}  ->  {path}")


if __name__ == "__main__":
    main()
