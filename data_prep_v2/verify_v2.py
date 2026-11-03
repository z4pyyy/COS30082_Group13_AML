"""Check DPD_Small_clean_v2 against the official CSVs and the v2 rules, then smoke-test the loader.

Run from the repo root:
    python data_prep_v2/verify_v2.py --raw ~/Downloads/DPD_Small --data ~/Downloads/DPD_Small_clean_v2
Exits with an AssertionError on the first failed check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

COLS = ["filename", "crop_id", "disease_id", "comp_id", "crop", "disease", "comp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    raw, data = args.raw.expanduser(), args.data.expanduser()

    off = {s: pd.read_csv(raw / f"DPD_Small_{s}.csv", header=None, names=COLS) for s in ["train", "test_seen", "test_unseen"]}
    d = {s: pd.read_csv(data / f"{s}.csv") for s in ["train", "val", "val_unseen", "test_seen", "test_unseen"]}
    maps = json.loads((data / "label_maps.json").read_text())
    stats = json.loads((data / "stats.json").read_text())

    # 1. official test sets unchanged (same files, same labels)
    for s in ["test_seen", "test_unseen"]:
        a = off[s].sort_values("filename").reset_index(drop=True)[COLS]
        b = d[s].sort_values("filename").reset_index(drop=True)[COLS]
        assert a.equals(b), f"{s} differs from the official CSV"
    print("ok  test_seen / test_unseen identical to the official CSVs")

    # 2. train + val + val_unseen = official train minus the 5 v1 exclusions
    ours = pd.concat([d["train"], d["val"], d["val_unseen"]])
    excluded = {e["filename"] for e in stats["excluded"]}
    assert len(excluded) == 5 and set(ours.filename) == set(off["train"].filename) - excluded
    print("ok  train + val + val_unseen = official train minus 5 near-duplicates of test images")

    # 3. no image in two splits
    names = pd.concat([df.filename for df in d.values()])
    assert names.is_unique, "an image is in two splits"
    print("ok  no image in two splits")

    # 4. held-out pairs: absent from train/val, crop and disease still present in train, not unseen-test pairs
    held = set(d["val_unseen"].comp)
    assert held.isdisjoint(d["train"].comp) and held.isdisjoint(d["val"].comp)
    assert set(d["val_unseen"].crop) <= set(d["train"].crop) and set(d["val_unseen"].disease) <= set(d["train"].disease)
    assert held.isdisjoint(d["test_unseen"].comp) and set(d["val_unseen"].disease).isdisjoint(d["test_unseen"].disease)
    assert sorted(d["val_unseen"].comp_id.unique()) == maps["heldout_comp_ids"]
    assert d["test_seen"].heldout_pair.sum() == stats["test_seen_images_of_heldout_pairs"] == \
        d["test_seen"].comp.isin(held).sum()
    print(f"ok  val_unseen: {len(held)} pairs, {len(d['val_unseen'])} images; crops and diseases all still in train")

    # 5. every unseen-test crop and disease is in train
    assert set(d["test_unseen"].crop) <= set(d["train"].crop) and set(d["test_unseen"].disease) <= set(d["train"].disease)
    print("ok  every test_unseen crop and disease appears in train")

    # 6. label maps consistent with the CSVs
    c2cd = {int(k): v for k, v in maps["comp_to_crop_disease"].items()}
    for df in d.values():
        for r in df[["comp_id", "crop_id", "disease_id"]].drop_duplicates().itertuples():
            assert c2cd[r.comp_id] == [r.crop_id, r.disease_id]
    print("ok  label_maps.json matches every CSV")

    # 7. images exist and are 224x224 RGB
    paths = pd.concat([df.path for df in d.values()])
    for p in paths:
        with Image.open(data / p) as im:
            assert im.size == (224, 224) and im.mode == "RGB", p
    print(f"ok  {len(paths)} images present, all 224x224 RGB")

    # 8. loader smoke test
    sys.path.insert(0, str(data.parent))
    mod = __import__(f"{data.name}.dataset", fromlist=["get_loaders"])
    loaders = mod.get_loaders(batch_size=8, num_workers=0)
    for s, dl in loaders.items():
        x, y = next(iter(dl))
        assert x.shape[1:] == (3, 224, 224) and set(y) == {"crop", "disease", "comp"}, s
        assert len(dl.dataset) == len(d[s]), s
    w = mod.class_weights(col="disease_id")
    w1 = mod.class_weights(d["train"], "disease_id", maps["num_diseases"])   # v1-style call
    assert (w - w1).abs().max() < 1e-6
    assert w.isfinite().all() and len(w) == maps["num_diseases"]
    cd = mod.comp_index()
    assert cd.shape == (maps["num_comps"], 2)
    print("ok  loader: every split yields (B,3,224,224) + {crop, disease, comp}; class_weights and comp_index work")
    print({s: len(dl.dataset) for s, dl in loaders.items()})


if __name__ == "__main__":
    main()
