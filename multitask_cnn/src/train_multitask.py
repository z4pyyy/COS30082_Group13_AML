"""Train the multi-task CNN baseline (brief 3.1, Fig. 1 right): ConvNeXt-Tiny + crop head + disease head.

Default = plain baseline, no tuning:
  * ImageNet-pretrained ConvNeXt-Tiny, backbone FROZEN (no fine-tuning); only the two linear heads train.
  * loss = CE(crop) + CE(disease), unweighted; no class re-weighting, no balanced sampler.
  * AdamW lr 1e-3, wd 0.05, cosine schedule, 15 epochs, batch 64, mixed precision on GPU.
  * team loader from dataset/DPD_Small_clean/dataset.py (its standard train augmentation).
  * checkpoint kept by best val composition Top-1 (val only, never test).
--finetune unfreezes the backbone (backbone lr = lr/10) for the later, tuned version.

Run from the project root:
    python -m src.train_multitask                       # frozen baseline
    python -m src.train_multitask --finetune --run convnext_mt_finetune
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import torch
import torch.nn as nn

from src.common import RUNS_DIR, comp_index, get_loaders, load_label_maps, save_json, seed_everything
from src.metrics import comp_scores, harmonic_mean, topk_correct
from src.models import MultiTaskCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="convnext_mt_baseline")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--finetune", action="store_true", help="unfreeze backbone (default: frozen)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10, help="early stopping on val_H")
    p.add_argument("--limit-batches", type=int, default=0, help="debug: stop each epoch after N batches")
    return p.parse_args()


@torch.no_grad()
def evaluate_val(model, loader, cd, device, limit=0):
    model.eval()
    n = crop_ok = dis_ok = 0
    comp_ok = {1: 0, 5: 0}
    for i, (x, y) in enumerate(loader):
        if limit and i >= limit:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x)
        yc, yd, yk = (y[k].to(device) for k in ("crop", "disease", "comp"))
        crop_ok += int((out["crop"].argmax(1) == yc).sum())
        dis_ok += int((out["disease"].argmax(1) == yd).sum())
        for k, v in topk_correct(comp_scores(out["crop"], out["disease"], cd), yk).items():
            comp_ok[k] += v
        n += x.size(0)
    return {"crop_acc": crop_ok / n, "disease_acc": dis_ok / n, "comp_top1": comp_ok[1] / n, "comp_top5": comp_ok[5] / n}


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RUNS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args) | {"device": str(device)}, out_dir / "config.json")

    maps = load_label_maps()
    cd = comp_index(maps).to(device)
    loaders = get_loaders(batch_size=args.batch_size, num_workers=args.workers)

    model = MultiTaskCNN(maps["num_crops"], maps["num_diseases"], args.backbone,
                         pretrained=True, freeze_backbone=not args.finetune).to(device)
    head_params = list(model.crop_head.parameters()) + list(model.disease_head.parameters())
    groups = [{"params": head_params, "lr": args.lr}]
    if args.finetune:
        groups.append({"params": model.backbone.parameters(), "lr": args.lr / 10})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = args.limit_batches or len(loaders["train"])
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}  trainable params={n_train/1e6:.2f}M  backbone {'fine-tuned' if args.finetune else 'frozen'}")

    best, bad, log_rows = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.finetune:
            model.backbone.eval()  # frozen: keep it in inference mode (no stochastic depth)
        t0, run_loss, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(loaders["train"]):
            if args.limit_batches and i >= args.limit_batches:
                break
            x = x.to(device, non_blocking=True)
            yc, yd = y["crop"].to(device), y["disease"].to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = ce(out["crop"], yc) + ce(out["disease"], yd)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
        val = evaluate_val(model, loaders["val"], cd, device, args.limit_batches)
        val_u = evaluate_val(model, loaders["val_unseen"], cd, device, args.limit_batches)
        h = harmonic_mean(val["comp_top1"], val_u["comp_top1"])
        row = {"epoch": epoch, "train_loss": run_loss / seen, **val,
               **{f"unseen_{k}": v for k, v in val_u.items()}, "val_H": h, "secs": round(time.time() - t0, 1)}
        log_rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val,
                 "val_unseen": val_u, "val_H": h}
        torch.save(state, out_dir / "last.pt")
        if h > best:
            best, bad = h, 0
            torch.save(state, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (no new best val_H for {args.patience} epochs)")
                break

    with open(out_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)
    print(f"best val_H (pair top-1) = {best:.4f}  ->  {out_dir / 'best.pt'}")
    print(f"next: python -m src.evaluate --run {args.run}")


if __name__ == "__main__":  # required on Windows (DataLoader workers)
    main()
