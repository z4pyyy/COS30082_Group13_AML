"""Step 1-4: audit DPD-Small images and CSVs.

Output (data_prep/reports/):
  image_stats.csv  one row per image: size, mode, real format, decode error, perceptual hash
  near_dups.csv    image pairs with pHash distance <= NEAR_DUP_DIST
  audit_summary.txt
Run: python data_prep/01_audit.py
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "DPD_Small"
IMG_DIR = RAW / "DPD_Small_dataset"
OUT = ROOT / "data_prep" / "reports"
COLS = ["filename", "crop_id", "disease_id", "comp_id", "crop", "disease", "comp"]
SPLITS = {"train": "DPD_Small_train.csv", "test_seen": "DPD_Small_test_seen.csv",
          "test_unseen": "DPD_Small_test_unseen.csv"}
NEAR_DUP_DIST = 4  # pHash Hamming distance (0-64); <=4 is visually the same image


def load_csvs() -> pd.DataFrame:
    frames = [pd.read_csv(RAW / f, header=None, names=COLS).assign(split=s) for s, f in SPLITS.items()]
    return pd.concat(frames, ignore_index=True)


def inspect(fname: str) -> dict:
    row = {"filename": fname, "width": None, "height": None, "mode": None,
           "real_format": None, "error": None, "phash": None}
    try:
        with Image.open(IMG_DIR / fname) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, real_format=im.format)
            im.load()  # full decode: catches truncated files
            row["phash"] = str(imagehash.phash(im.convert("RGB")))
    except Exception as e:  # noqa: BLE001 - record every failure
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def near_duplicates(stats: pd.DataFrame) -> pd.DataFrame:
    ok = stats.dropna(subset=["phash"]).reset_index(drop=True)
    bits = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in ok.phash], dtype=np.uint8)
    pairs = []
    for i in range(len(bits) - 1):
        dist = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(dist <= NEAR_DUP_DIST)[0]:
            pairs.append((i, i + 1 + j, int(dist[j])))
    a = ok.loc[[p[0] for p in pairs]].reset_index(drop=True).add_suffix("_a")
    b = ok.loc[[p[1] for p in pairs]].reset_index(drop=True).add_suffix("_b")
    out = pd.concat([a, b], axis=1)
    out["distance"] = [p[2] for p in pairs]
    keep = ["filename", "split", "comp"]
    return out[[f"{k}_{s}" for s in "ab" for k in keep] + ["distance"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_csvs()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(inspect, meta.filename, chunksize=64))
    stats = meta.merge(pd.DataFrame(rows), on="filename")
    stats["ext"] = stats.filename.str.rsplit(".", n=1).str[1].str.lower().replace({"jpeg": "jpg"})
    # MPO = multi-picture JPEG from phone cameras; it is a normal JPEG for our use
    stats["ext_mismatch"] = stats.ext != stats.real_format.str.lower().replace({"jpeg": "jpg", "mpo": "jpg"})
    stats["source"] = stats.filename.str.split("_").str[0]
    stats["aspect"] = (stats.width / stats.height).round(3)
    stats.to_csv(OUT / "image_stats.csv", index=False)

    dups = near_duplicates(stats)
    dups.to_csv(OUT / "near_dups.csv", index=False)

    cross = dups[dups.split_a != dups.split_b]
    label_conflict = dups[dups.comp_a != dups.comp_b]
    lines = [
        f"images: {len(stats)}",
        f"decode errors: {stats.error.notna().sum()}",
        f"extension mismatch: {stats.ext_mismatch.sum()}",
        f"modes: {stats['mode'].value_counts().to_dict()}",
        f"width  min/median/max: {stats.width.min()}/{stats.width.median():.0f}/{stats.width.max()}",
        f"height min/median/max: {stats.height.min()}/{stats.height.median():.0f}/{stats.height.max()}",
        f"aspect (w/h) p5/median/p95: {stats.aspect.quantile(.05):.2f}/{stats.aspect.median():.2f}/{stats.aspect.quantile(.95):.2f}",
        f"smaller than 224 on short side: {(stats[['width', 'height']].min(axis=1) < 224).sum()}",
        f"near-dup pairs (pHash<= {NEAR_DUP_DIST}): {len(dups)}",
        f"  cross-split pairs: {len(cross)}  {cross.groupby(['split_a', 'split_b']).size().to_dict()}",
        f"  pairs with different labels: {len(label_conflict)}",
    ]
    (OUT / "audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
