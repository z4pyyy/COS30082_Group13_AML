"""Evaluate a trained multi-task run on test_seen and test_unseen (brief 3.3).

Outputs in runs/<run>/:
  metrics.json                 Top-1/Top-5 comp accuracy (seen, unseen), harmonic mean, crop/disease accuracy
  predictions_<split>.csv      top-5 predicted comps per image
  confusion_test_seen.csv      323 x 323 confusion matrix (rows = true seen comp, cols = predicted seen comp)
  confusion_test_seen.png      row-normalised heat map
  top_confusions_test_seen.csv most frequent true -> predicted mistakes (for the report)

    python -m src.evaluate --run convnext_mt_baseline
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.common import RUNS_DIR, comp_index, get_loaders, load_label_maps, save_json
from src.metrics import comp_scores, harmonic_mean, topk_correct
from src.models import MultiTaskCNN


@torch.no_grad()
def run_split(model, loader, cd, device):
    model.eval()
    tops, ys, crop_ok, dis_ok, n = [], [], 0, 0, 0
    hits = {1: 0, 5: 0}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x)
        s = comp_scores(out["crop"], out["disease"], cd)
        yk = y["comp"].to(device)
        for k, v in topk_correct(s, yk).items():
            hits[k] += v
        crop_ok += int((out["crop"].argmax(1).cpu() == y["crop"]).sum())
        dis_ok += int((out["disease"].argmax(1).cpu() == y["disease"]).sum())
        tops.append(s.topk(5, dim=1).indices.cpu())
        ys.append(y["comp"])
        n += x.size(0)
    return {"top1": hits[1] / n, "top5": hits[5] / n, "crop_acc": crop_ok / n, "disease_acc": dis_ok / n,
            "n": n}, torch.cat(tops).numpy(), torch.cat(ys).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="convnext_mt_baseline")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS_DIR / args.run
    state = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)
    maps = load_label_maps()
    cd = comp_index(maps).to(device)
    model = MultiTaskCNN(maps["num_crops"], maps["num_diseases"], state["args"]["backbone"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    loaders = get_loaders(batch_size=args.batch_size, num_workers=args.workers)

    results = {}
    for split in ("test_seen", "test_unseen"):
        m, top5, y = run_split(model, loaders[split], cd, device)
        results[split] = m
        names = maps["comp"]
        pd.DataFrame({"path": loaders[split].dataset.df["path"], "true": [names[int(c)] for c in y],
                      **{f"pred{i+1}": [names[int(c)] for c in top5[:, i]] for i in range(5)}}
                     ).to_csv(run_dir / f"predictions_{split}.csv", index=False)
        if split == "test_seen":
            seen_ids = list(maps["seen_comp_ids"])
            pos = {c: i for i, c in enumerate(seen_ids)}
            cm = np.zeros((len(seen_ids), len(seen_ids)), dtype=int)
            pred_unseen = 0
            for t, pr in zip(y, top5[:, 0]):
                if int(pr) in pos:
                    cm[pos[int(t)], pos[int(pr)]] += 1
                else:
                    pred_unseen += 1  # a seen image predicted as an unseen comp
            labels = [names[c] for c in seen_ids]
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(run_dir / "confusion_test_seen.csv")
            m["seen_images_predicted_as_unseen_comp"] = pred_unseen
            off = [(labels[i], labels[j], int(cm[i, j])) for i, j in zip(*np.nonzero(cm)) if i != j]
            pd.DataFrame(sorted(off, key=lambda r: -r[2])[:30], columns=["true", "predicted", "count"]
                         ).to_csv(run_dir / "top_confusions_test_seen.csv", index=False)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                fig, ax = plt.subplots(figsize=(10, 9))
                im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(f"Confusion matrix, test_seen ({len(seen_ids)} seen comps, row-normalised)")
                ax.set_xlabel("predicted comp id (seen order)")
                ax.set_ylabel("true comp id (seen order)")
                fig.colorbar(im, fraction=0.046)
                fig.tight_layout()
                fig.savefig(run_dir / "confusion_test_seen.png", dpi=150)
                plt.close(fig)
            except ImportError:
                print("matplotlib not installed: skipped the PNG")

    s, u = results["test_seen"], results["test_unseen"]
    results["harmonic_mean_top1"] = harmonic_mean(s["top1"], u["top1"])
    results["harmonic_mean_top5"] = harmonic_mean(s["top5"], u["top5"])
    save_json(results, run_dir / "metrics.json")
    print(f"{'':12}{'Top-1':>8}{'Top-5':>8}{'crop':>8}{'disease':>9}")
    for k in ("test_seen", "test_unseen"):
        r = results[k]
        print(f"{k:12}{r['top1']:8.4f}{r['top5']:8.4f}{r['crop_acc']:8.4f}{r['disease_acc']:9.4f}")
    print(f"harmonic mean  Top-1 {results['harmonic_mean_top1']:.4f}   Top-5 {results['harmonic_mean_top5']:.4f}")


if __name__ == "__main__":
    main()
