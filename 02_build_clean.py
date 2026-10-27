"""Step 5-7: build the clean dataset for training.

Input:  DPD_Small/ (raw images + 3 official CSVs), data_prep/reports/ (from 01_audit.py)
Output: DPD_Small_clean/
  images/<stem>.jpg      224x224 RGB JPEG, EXIF rotation applied
  train.csv val.csv test_seen.csv test_unseen.csv   (with header)
  manifest.csv           every raw image + split + flags + exclude reason
  label_maps.json        id <-> name for crop / disease / comp, comp -> (crop, disease)
  stats.json             class counts and dataset mean/std
Run: python data_prep/02_build_clean.py
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "DPD_Small" / "DPD_Small_dataset"
REPORTS = ROOT / "data_prep" / "reports"
OUT = ROOT / "DPD_Small_clean"
IMG_OUT = OUT / "images"

SIZE = 224
JPEG_QUALITY = 95
VAL_FRAC = 0.10
VAL_MIN_IMAGES = 5  # comps with fewer train images stay 100% in train
SEED = 42
LABEL_COLS = ["crop_id", "disease_id", "comp_id", "crop", "disease", "comp"]


ORIENTATION_OPS = {2: [Image.Transpose.FLIP_LEFT_RIGHT], 3: [Image.Transpose.ROTATE_180],
                   4: [Image.Transpose.FLIP_TOP_BOTTOM], 5: [Image.Transpose.TRANSPOSE],
                   6: [Image.Transpose.ROTATE_270], 7: [Image.Transpose.TRANSVERSE],
                   8: [Image.Transpose.ROTATE_90]}


def to_rgb(im: Image.Image) -> Image.Image:
    # Apply only the EXIF orientation tag. ImageOps.exif_transpose crashes on some malformed EXIF blocks.
    try:
        orientation = im.getexif().get(0x0112, 1)
    except Exception:  # noqa: BLE001 - unreadable EXIF = no rotation
        orientation = 1
    for op in ORIENTATION_OPS.get(orientation, []):
        im = im.transpose(op)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def resize_one(args: tuple[str, str]) -> tuple[str, list[float], list[float]]:
    src, dst = args
    with Image.open(SRC_DIR / src) as im:
        # Plain resize (no crop): keeps lesions at the image edge. Median aspect is 1.0.
        out = to_rgb(im).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    out.save(IMG_OUT / dst, "JPEG", quality=JPEG_QUALITY)
    px = np.asarray(out, dtype=np.float64) / 255.0
    return dst, px.mean(axis=(0, 1)).tolist(), (px ** 2).mean(axis=(0, 1)).tolist()


def dup_groups(files: pd.Series, pairs: pd.DataFrame) -> pd.Series:
    """Union-find: near-duplicate images get one group id, so they go to the same split."""
    parent = {f: f for f in files}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs.filename_a, pairs.filename_b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return files.map(find)


def make_val_split(train: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(SEED)
    group = dup_groups(train.filename, pairs)
    split = pd.Series("train", index=train.index)
    for _, idx in train.groupby("comp_id").groups.items():
        if len(idx) < VAL_MIN_IMAGES:
            continue
        n_val = max(1, round(len(idx) * VAL_FRAC))
        groups = rng.permutation(group[idx].unique())
        taken = 0
        for g in groups:
            members = idx[group[idx] == g]
            if taken + len(members) > n_val:
                continue
            split[members] = "val"
            taken += len(members)
            if taken == n_val:
                break
    return split


def main() -> None:
    stats = pd.read_csv(REPORTS / "image_stats.csv")
    pairs = pd.read_csv(REPORTS / "near_dups.csv")
    assert stats.error.isna().all(), "decode errors found: fix before build"

    m = stats.copy()
    m["exclude_reason"] = ""
    leak = pairs[(pairs.split_a == "train") & (pairs.split_b.str.startswith("test"))].filename_a
    leak = pd.concat([leak, pairs[(pairs.split_b == "train") & (pairs.split_a.str.startswith("test"))].filename_b])
    m.loc[m.filename.isin(leak), "exclude_reason"] = "near_duplicate_of_test_image"

    in_train = (m.split == "train") & (m.exclude_reason == "")
    m.loc[in_train, "split"] = make_val_split(m[in_train], pairs)
    m.loc[m.exclude_reason != "", "split"] = "excluded"

    m["path"] = "images/" + m.filename.str.rsplit(".", n=1).str[0] + ".jpg"
    assert m.path.is_unique, "two raw files map to one output name"
    near = set(pairs.filename_a) | set(pairs.filename_b)
    m["flag_near_dup"] = m.filename.isin(near)
    m["flag_upscaled"] = m[["width", "height"]].min(axis=1) < SIZE
    m["real_format"] = m.real_format.replace({"MPO": "JPEG"})

    IMG_OUT.mkdir(parents=True, exist_ok=True)
    keep = m[m.split != "excluded"]
    jobs = list(zip(keep.filename, keep.path.str.removeprefix("images/")))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(resize_one, jobs, chunksize=64))

    train_names = set(keep[keep.split == "train"].path.str.removeprefix("images/"))
    mean_sq = np.array([[r[1], r[2]] for r in res if r[0] in train_names])
    mean = mean_sq[:, 0].mean(axis=0)
    std = np.sqrt(mean_sq[:, 1].mean(axis=0) - mean ** 2)

    cols = ["path", "filename", *LABEL_COLS, "source", "flag_near_dup", "flag_upscaled"]
    for s in ["train", "val", "test_seen", "test_unseen"]:
        m[m.split == s][cols].to_csv(OUT / f"{s}.csv", index=False)
    m[["path", "filename", "split", "exclude_reason", *LABEL_COLS, "source", "width", "height",
       "mode", "real_format", "ext_mismatch", "flag_near_dup", "flag_upscaled"]].to_csv(OUT / "manifest.csv", index=False)

    def id_map(id_col: str, name_col: str) -> dict[int, str]:
        return {int(k): v for k, v in m.drop_duplicates(id_col).sort_values(id_col)[[id_col, name_col]].values}

    comps = m.drop_duplicates("comp_id").sort_values("comp_id")
    unseen = sorted(m[m.split == "test_unseen"].comp_id.unique().tolist())
    label_maps = {
        "num_crops": int(m.crop_id.nunique()), "num_diseases": int(m.disease_id.nunique()),
        "num_comps": int(m.comp_id.nunique()),
        "crop": id_map("crop_id", "crop"), "disease": id_map("disease_id", "disease"), "comp": id_map("comp_id", "comp"),
        "comp_to_crop_disease": {int(r.comp_id): [int(r.crop_id), int(r.disease_id)] for r in comps.itertuples()},
        "seen_comp_ids": sorted(set(comps.comp_id) - set(unseen)), "unseen_comp_ids": unseen,
    }
    (OUT / "label_maps.json").write_text(json.dumps(label_maps, indent=1), encoding="utf-8")

    tr = m[m.split == "train"]
    summary = {
        "image_size": SIZE, "seed": SEED,
        "split_sizes": m.split.value_counts().to_dict(),
        "train_mean_rgb": mean.round(4).tolist(), "train_std_rgb": std.round(4).tolist(),
        "imagenet_mean_rgb": [0.485, 0.456, 0.406], "imagenet_std_rgb": [0.229, 0.224, 0.225],
        "train_counts": {k: {int(i): int(c) for i, c in tr[k].value_counts().sort_index().items()}
                         for k in ["crop_id", "disease_id", "comp_id"]},
        "excluded": m[m.split == "excluded"][["filename", "exclude_reason"]].to_dict("records"),
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("split sizes:", summary["split_sizes"])
    print("train mean/std:", summary["train_mean_rgb"], summary["train_std_rgb"])
    print("val comps:", m[m.split == "val"].comp_id.nunique(), "| train comps:", tr.comp_id.nunique(),
          "| train crops/diseases:", tr.crop_id.nunique(), tr.disease_id.nunique())


if __name__ == "__main__":
    main()
