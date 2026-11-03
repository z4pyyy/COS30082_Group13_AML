"""Standard DPD_Small_clean_v2 outputs for one multitask run (same files and schema as the independent tracks).

    python -m src.evaluate_v2 --run convnext_mt_finetune_v2 --name ConvNeXt-Tiny-finetune --out output/multitask_v2

Reads runs/<run>/best.pt. Writes into --out (adds to results.json / summary.csv if they already exist):
  <name>_<split>_confusion_matrix.csv/.png, <name>_predictions_<split>.csv            crop+disease pair (333)
  <name>_crop_<split>_confusion_matrix..., <name>_disease_<split>_confusion_matrix... the two heads
  results.json, summary.csv
split = val, val_unseen, seen (test_seen), unseen (test_unseen).
Needs src/common.py pointing at DPD_Small_clean_v2 (see HANDOFF_v2.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

from src.common import DATA_ROOT, RUNS_DIR, comp_index, get_loaders, load_label_maps
from src.metrics import comp_scores
from src.models import MultiTaskCNN

sys.path.insert(0, str(DATA_ROOT))
from metrics import evaluate_split, harmonic_dict, topk_acc  # noqa: E402  (DPD_Small_clean_v2/metrics.py)

SPLITS = {"val": "val", "val_unseen": "val_unseen", "seen": "test_seen", "unseen": "test_unseen"}


@torch.no_grad()
def collect(model, loader, cd, device, limit=0):
    model.eval()
    s, c, d, ys = [], [], [], {"comp": [], "crop": [], "disease": []}
    for i, (x, y) in enumerate(loader):
        if limit and i >= limit:
            break
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        s.append(comp_scores(out["crop"], out["disease"], cd).float().cpu())
        c.append(out["crop"].float().cpu()); d.append(out["disease"].float().cpu())
        for k in ys:
            ys[k].append(y[k])
    return torch.cat(s), torch.cat(c), torch.cat(d), {k: torch.cat(v) for k, v in ys.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder in runs/ holding best.pt")
    p.add_argument("--name", required=True, help="row name, e.g. ConvNeXt-Tiny-finetune")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-batches", type=int, default=0, help="smoke tests only")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(RUNS_DIR / a.run / "best.pt", map_location=dev, weights_only=False)
    maps = load_label_maps()
    cd = comp_index(maps).to(dev)
    model = MultiTaskCNN(maps["num_crops"], maps["num_diseases"], st["args"]["backbone"], pretrained=False).to(dev)
    model.load_state_dict(st["model"])
    loaders = get_loaders(batch_size=64, num_workers=a.workers)
    names = {k: [maps[k][i] for i in range(n)] for k, n in
             [("comp", maps["num_comps"]), ("crop", maps["num_crops"]), ("disease", maps["num_diseases"])]}

    r = {"backbone": st["args"]["backbone"], "finetune": bool(st["args"].get("finetune")), "run": a.run,
         "best_epoch": st.get("epoch"), "selection": "max H(val, val_unseen) pair Top-1",
         "pair": {}, "crop_head": {}, "disease_head": {}}
    for key, split in SPLITS.items():
        s, c, d, y = collect(model, loaders[split], cd, dev, a.limit_batches)
        df = loaders[split].dataset.df.iloc[: len(s)]
        paths = df.path.tolist()
        heads = {"crop_head_pred": [names["crop"][i] for i in c.argmax(1)],
                 "disease_head_pred": [names["disease"][i] for i in d.argmax(1)]}
        r["pair"][key] = evaluate_split(s, y["comp"], names["comp"], a.out, a.name, key, paths, extra_cols=heads)
        r["crop_head"][key] = evaluate_split(c, y["crop"], names["crop"], a.out, f"{a.name}_crop", key, paths)
        r["disease_head"][key] = evaluate_split(d, y["disease"], names["disease"], a.out, f"{a.name}_disease", key, paths)
        if key == "seen":
            keep = torch.tensor(~df.heldout_pair.values)
            r["pair"]["seen_excl_heldout"] = topk_acc(s[keep], y["comp"][keep])
            r["crop_head"]["seen_excl_heldout"] = topk_acc(c[keep], y["crop"][keep])
            r["disease_head"]["seen_excl_heldout"] = topk_acc(d[keep], y["disease"][keep])
    for t in ("pair", "crop_head", "disease_head"):
        r[t]["harmonic"] = harmonic_dict(r[t]["seen"], r[t]["unseen"])

    res_path = os.path.join(a.out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    results[a.name] = r
    json.dump(results, open(res_path, "w"), indent=2)

    rows = []
    for name, x in results.items():
        pr = x["pair"]
        row = {"Model": name, "Val Top-1": pr["val"]["1"], "Val-unseen Top-1": pr["val_unseen"]["1"]}
        for k in ("1", "3", "5"):
            row |= {f"Seen Top-{k}": pr["seen"][k], f"Unseen Top-{k}": pr["unseen"][k], f"H Top-{k}": pr["harmonic"][k]}
        row |= {"Seen excl. held-out Top-1": pr["seen_excl_heldout"]["1"],
                "Seen per-class Top-1": pr["seen"]["per_class_top1"], "Unseen per-class Top-1": pr["unseen"]["per_class_top1"],
                "Crop head seen": x["crop_head"]["seen"]["1"], "Crop head unseen": x["crop_head"]["unseen"]["1"],
                "Disease head seen": x["disease_head"]["seen"]["1"], "Disease head unseen": x["disease_head"]["unseen"]["1"]}
        rows.append(row)
    s = pd.DataFrame(rows).round(2)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
