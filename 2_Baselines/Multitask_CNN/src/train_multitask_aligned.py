"""Multi-task baseline trained with EXACTLY the independent tracks' recipe (DPD_Small_clean shared protocol).

Why this exists: the earlier multitask runs differed from the independent crop/disease runs in more than "one
network vs two": torchvision weights, no label smoothing, cosine without warm-up, head lr 10x the backbone lr,
batch 32, ConvNeXt only. Any gap between the tracks mixed those differences with the multitask effect.
This script changes only what the tracks need to share:

  * backbones/pretrained tags/learning rates identical to 2_Baselines/train_independent.py (timm, ImageNet-1k)
  * AdamW wd 0.05, OneCycleLR pct_start 0.1, label smoothing 0.1 on BOTH heads, one lr for the whole network
  * effective batch 64 (use --accum 2 for a 4 GB GPU: micro-batch 32 x 2 steps; exact for ConvNeXt, near-exact for BN nets)
  * best epoch = max H(val, val_unseen) pair Top-1, patience 10, seed 42, 30 epochs
  * loss = CE(crop) + w_disease * CE(disease); w_disease = 1 for the main run

    python -m src.train_multitask_aligned --models ConvNeXt-Tiny --accum 2          # laptop
    python -m src.train_multitask_aligned                                           # Colab/Kaggle: all 3 backbones
    python -m src.evaluate --run mt_aligned_ConvNeXt-Tiny --name ConvNeXt-Tiny-MT --out output/multitask_aligned
"""
from __future__ import annotations

import argparse
import csv
import time

import torch
import torch.nn as nn

from src.common import RUNS_DIR, comp_index, get_loaders, load_label_maps, save_json, seed_everything
from src.metrics import harmonic_mean
from src.models import MultiTaskCNN
from src.train_multitask import evaluate_val

MODELS = {  # name -> (timm id with pinned ImageNet-1k tag, lr): same table as train_independent.py
    "ResNet50": ("resnet50.tv2_in1k", 3e-4),
    "EfficientNet-B0": ("efficientnet_b0.ra_in1k", 5e-4),
    "ConvNeXt-Tiny": ("convnext_tiny.fb_in1k", 1e-4),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--run-prefix", default="mt_aligned")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64, help="EFFECTIVE batch size")
    p.add_argument("--accum", type=int, default=1, help="gradient-accumulation steps (micro-batch = batch-size / accum)")
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--w-disease", type=float, default=1.0, help="main run: 1.0; other values = separate, named runs")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-batches", type=int, default=0, help="smoke test only")
    return p.parse_args()


def main():
    a = parse_args()
    assert a.batch_size % a.accum == 0, "--batch-size must be divisible by --accum"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    maps = load_label_maps()
    cd = comp_index(maps).to(dev)
    loaders = get_loaders(batch_size=a.batch_size // a.accum, num_workers=a.workers)
    lim = a.limit_batches

    for name in a.models:
        timm_id, lr = MODELS[name]
        out_dir = RUNS_DIR / f"{a.run_prefix}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(a.seed)
        backbone = f"timm:{timm_id}"
        # evaluate reads args["backbone"] / args["finetune"] from the checkpoint
        cfg = vars(a) | {"backbone": backbone, "finetune": True, "lr": lr, "device": str(dev),
                         "torch": torch.__version__, "optimizer": "AdamW", "schedule": "OneCycleLR pct_start=0.1"}
        save_json(cfg, out_dir / "config.json")

        model = MultiTaskCNN(maps["num_crops"], maps["num_diseases"], backbone, pretrained=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.weight_decay)
        n_batches = lim or len(loaders["train"])
        steps = a.epochs * (-(-n_batches // a.accum))            # optimizer steps (ceil per epoch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ce = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)
        print(f"\n===== {name} ({timm_id}) lr={lr} micro-batch={a.batch_size // a.accum} x accum={a.accum} =====", flush=True)

        best, best_ep, bad, hist = -1.0, 0, 0, []
        for ep in range(1, a.epochs + 1):
            model.train()
            t0, tl, n = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, (x, y) in enumerate(loaders["train"]):
                if lim and i >= lim:
                    break
                x = x.to(dev, non_blocking=True)
                yc, yd = y["crop"].to(dev), y["disease"].to(dev)
                with torch.autocast(dev.type, enabled=amp):
                    out = model(x)
                    loss = ce(out["crop"], yc) + a.w_disease * ce(out["disease"], yd)
                scaler.scale(loss / a.accum).backward()
                tl += loss.item() * x.size(0); n += x.size(0)
                last = (i + 1 == n_batches) or ((i + 1) % a.accum == 0)
                if last:
                    scaler.step(opt); scaler.update(); sched.step()
                    opt.zero_grad(set_to_none=True)
            val = evaluate_val(model, loaders["val"], cd, dev, lim)
            val_u = evaluate_val(model, loaders["val_unseen"], cd, dev, lim)
            h = harmonic_mean(val["comp_top1"], val_u["comp_top1"])
            row = {"epoch": ep, "train_loss": tl / n, **val, **{f"unseen_{k}": v for k, v in val_u.items()},
                   "val_H": h, "secs": round(time.time() - t0, 1)}
            hist.append(row)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            state = {"model": model.state_dict(), "args": cfg, "epoch": ep, "val": val, "val_unseen": val_u, "val_H": h}
            torch.save(state, out_dir / "last.pt")
            if h > best:
                best, best_ep, bad = h, ep, 0
                torch.save(state, out_dir / "best.pt")
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"early stop at epoch {ep}; best epoch {best_ep} (val_H {best:.4f})")
                    break
        with open(out_dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
        print(f"[{name}] best val_H = {best:.4f} at epoch {best_ep} -> {out_dir / 'best.pt'}")
        print(f"next: python -m src.evaluate --run {a.run_prefix}_{name} --name {name}-MT --out output/multitask_aligned")
        del model, opt
        if amp:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
