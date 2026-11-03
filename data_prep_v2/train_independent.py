"""Reference training script for the two INDEPENDENT tracks (crop CNN, disease CNN) on DPD_Small_clean_v2.
Follows the shared protocol in README.md and writes the standard output folder.

Kaggle / Colab (GPU):
    !pip -q install timm
    !python /kaggle/input/<slug>/train_independent.py --task crop    --out /kaggle/working/crop_independent_v2
    !python /kaggle/input/<slug>/train_independent.py --task disease --out /kaggle/working/disease_independent_v2
Smoke test on CPU (a few batches, random weights):
    python train_independent.py --task crop --out /tmp/smoke --models ResNet50 --epochs 2 --limit-batches 2 --no-pretrained

Output (in --out): config.json, results.json, summary.csv, <Model>_history.csv, <Model>_curves.png,
<Model>_<split>_confusion_matrix.csv/.png and <Model>_predictions_<split>.csv for split in val, val_unseen, seen, unseen,
all_models_accuracy.png, harmonic_mean.png, <Model>_best.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DPDDataset, build_transforms, get_loaders, load_label_maps  # noqa: E402
from metrics import evaluate_split, harmonic, harmonic_dict, topk_acc  # noqa: E402

# name -> (timm id with a pinned ImageNet-1k pretrained tag, learning rate)
MODELS = {
    "ResNet50": ("resnet50.tv2_in1k", 3e-4),
    "EfficientNet-B0": ("efficientnet_b0.ra_in1k", 5e-4),
    "ConvNeXt-Tiny": ("convnext_tiny.fb_in1k", 1e-4),
}
EVAL_SPLITS = {"val": "val", "val_unseen": "val_unseen", "seen": "test_seen", "unseen": "test_unseen"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["crop", "disease"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    p.add_argument("--limit-batches", type=int, default=0, help="smoke tests only: stop every loop after N batches")
    return p.parse_args()


def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    import timm
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    maps = load_label_maps()
    n_cls = maps["num_crops"] if a.task == "crop" else maps["num_diseases"]
    names = [maps[a.task][i] for i in range(n_cls)]
    loaders = get_loaders(batch_size=a.batch_size, num_workers=a.workers)
    train_eval = torch.utils.data.DataLoader(DPDDataset("train", transform=build_transforms(False)),
                                             batch_size=a.batch_size * 2, num_workers=a.workers)
    lim = a.limit_batches
    print("split sizes:", {s: len(dl.dataset) for s, dl in loaders.items()}, "| device:", dev, flush=True)

    def batches(loader):
        for i, b in enumerate(loader):
            if lim and i >= lim:
                return
            yield b

    def ds_df(loader):  # dataframe rows that predict() actually saw (all of them unless --limit-batches)
        df = loader.dataset.df
        return df.iloc[: lim * loader.batch_size] if lim else df

    @torch.no_grad()
    def predict(model, loader):
        model.eval(); outs, ys = [], []
        for x, y in batches(loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outs.append(model(x.to(dev, non_blocking=True)).float().cpu())
            ys.append(y[a.task])
        return torch.cat(outs), torch.cat(ys)

    config = vars(a) | {
        "dataset": "DPD_Small_clean_v2", "protocol": "best epoch = max H(val Top-1, val_unseen Top-1)",
        "models": {m: {"timm_id": MODELS[m][0], "lr": MODELS[m][1]} for m in a.models},
        "optimizer": "AdamW", "schedule": "OneCycleLR pct_start=0.1", "amp": amp, "device": dev,
        "gpu": torch.cuda.get_device_name(0) if amp else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__},
    }
    cfg_path = f"{a.out}/config.json"
    if os.path.exists(cfg_path):  # rerun into the same folder: keep earlier backbones' entries
        old = json.load(open(cfg_path))
        config["models"] = old.get("models", {}) | config["models"]
    json.dump(config, open(cfg_path, "w"), indent=2)

    res_path = f"{a.out}/results.json"
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}   # rerun of missing backbones keeps the rest
    for name in a.models:
        timm_id, lr = MODELS[name]
        print(f"\n===== {name} ({timm_id}) =====")
        seed_all(a.seed)
        model = timm.create_model(timm_id, pretrained=not a.no_pretrained, num_classes=n_cls).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.weight_decay)
        steps = a.epochs * (lim or len(loaders["train"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        crit = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ckpt = f"{a.out}/{name}_best.pt"
        best, best_ep, bad, hist = -1.0, 0, 0, []

        for ep in range(1, a.epochs + 1):
            model.train(); tl = tc = n = 0; t0 = time.time()
            for x, y in batches(loaders["train"]):
                x, y = x.to(dev, non_blocking=True), y[a.task].to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = model(x); loss = crit(out, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
            lv, yv = predict(model, loaders["val"])
            lu, yu = predict(model, loaders["val_unseen"])
            va, vu = topk_acc(lv, yv)["1"], topk_acc(lu, yu)["1"]
            h = harmonic(va, vu)
            hist.append(dict(epoch=ep, train_loss=tl / n, train_acc_aug=100 * tc / n,
                             val_loss=nn.functional.cross_entropy(lv, yv).item(), val_acc=va,
                             val_unseen_acc=vu, val_H=h, secs=round(time.time() - t0, 1)))
            print(f"[{name}] ep {ep:02d} | train loss {tl/n:.4f} acc(aug) {100*tc/n:.2f} | "
                  f"val {va:.2f} | val_unseen {vu:.2f} | H {h:.2f} | {time.time()-t0:.0f}s", flush=True)
            if h > best:
                best, best_ep, bad = h, ep, 0
                torch.save({"state_dict": model.state_dict(), "timm_id": timm_id, "num_classes": n_cls,
                            "class_names": names, "task": a.task, "epoch": ep, "val_H": h, "val_top1": va,
                            "val_unseen_top1": vu, "dataset": "DPD_Small_clean_v2"}, ckpt)
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"[{name}] early stop at epoch {ep}; best epoch {best_ep} (H {best:.2f})")
                    break

        hist = pd.DataFrame(hist)
        hist.to_csv(f"{a.out}/{name}_history.csv", index=False)
        plot_curves(name, hist, best_ep, a.out)
        model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])

        r = {"timm_id": timm_id, "lr": lr, "best_epoch": best_ep, "epochs_run": len(hist),
             "selection": "max H(val, val_unseen) Top-1"}
        lg, yy = predict(model, train_eval)
        r["train_clean_top1"] = topk_acc(lg, yy)["1"]
        for key, split in EVAL_SPLITS.items():
            lg, yy = predict(model, loaders[split])
            df = ds_df(loaders[split])
            r[key] = evaluate_split(lg, yy, names, a.out, name, key, paths=df.path.tolist())
            if key == "seen":
                keep = torch.tensor(~df.heldout_pair.values)
                r["seen_excl_heldout"] = topk_acc(lg[keep], yy[keep])
        r["harmonic"] = harmonic_dict(r["seen"], r["unseen"])
        results[name] = r
        json.dump(results, open(f"{a.out}/results.json", "w"), indent=2)   # after every model: a crash keeps the rest
        print(json.dumps({k: r[k] for k in ("best_epoch", "seen", "unseen", "harmonic")}, indent=1))
        del model; torch.cuda.empty_cache()

    write_summary(results, a.out)


def plot_curves(name: str, h: pd.DataFrame, best_ep: int, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h.epoch, h.train_acc_aug, label="Train (augmented batches)")
    ax[0].plot(h.epoch, h.val_acc, label="val (seen pairs)")
    ax[0].plot(h.epoch, h.val_unseen_acc, label="val_unseen (held-out pairs)")
    ax[0].plot(h.epoch, h.val_H, "k--", label="H(val, val_unseen)")
    ax[1].plot(h.epoch, h.train_loss, label="Train loss")
    ax[1].plot(h.epoch, h.val_loss, label="val loss")
    for x in ax:
        x.axvline(best_ep, ls=":", c="gray"); x.grid(alpha=.3); x.legend(fontsize=8); x.set_xlabel("Epoch")
    ax[0].set_title(f"{name}: accuracy (%), best epoch {best_ep}"); ax[1].set_title(f"{name}: loss")
    fig.tight_layout(); fig.savefig(f"{out}/{name}_curves.png", dpi=150); plt.close(fig)


def write_summary(results: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for m, r in results.items():
        row = {"Model": m, "Val Top-1": r["val"]["1"], "Val-unseen Top-1": r["val_unseen"]["1"]}
        for k in ("1", "3", "5"):
            row |= {f"Seen Top-{k}": r["seen"][k], f"Unseen Top-{k}": r["unseen"][k], f"H Top-{k}": r["harmonic"][k]}
        row |= {"Seen excl. held-out Top-1": r["seen_excl_heldout"]["1"],
                "Seen per-class Top-1": r["seen"]["per_class_top1"],
                "Unseen per-class Top-1": r["unseen"]["per_class_top1"], "Train (clean) Top-1": r["train_clean_top1"]}
        rows.append(row)
    s = pd.DataFrame(rows).round(2)
    s.to_csv(f"{out}/summary.csv", index=False)
    x = np.arange(len(s))
    for fname, cols in [("all_models_accuracy.png", ["Seen Top-1", "Unseen Top-1", "Seen Top-5", "Unseen Top-5"]),
                        ("harmonic_mean.png", ["H Top-1", "H Top-3", "H Top-5"])]:
        fig, ax = plt.subplots(figsize=(8, 4)); w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            b = ax.bar(x + i * w, s[c], w, label=c); ax.bar_label(b, fmt="%.1f", fontsize=7)
        ax.set_xticks(x + w * (len(cols) - 1) / 2, s.Model); ax.set_ylabel("%"); ax.set_ylim(0, 105)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(f"{out}/{fname}", dpi=150)
        plt.close(fig)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
