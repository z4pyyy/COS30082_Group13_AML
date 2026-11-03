"""Build DPD_Small_clean_v2 from the raw DPD_Small images.

v2 = v1 (same cleaning, same pixels, same official test sets) plus:
  * val_unseen.csv: 6 seen crop+disease pairs held out of train/val, so models can be tuned for unseen pairs
    (each held-out pair's crop and disease still appear in train on other pairs);
  * extra metadata columns (source / crop-disease spread) and a `heldout_pair` flag on test_seen;
  * a standalone loader (dataset.py) and docs copied into the output folder.

Reuses v1's functions from 02_build_clean.py (EXIF/RGB handling, near-duplicate-aware val split) and the
v1 audit outputs in reports/ (image_stats.csv, near_dups.csv), so the v1 part is reproduced exactly.

Run from the repo root:
    python data_prep_v2/build_v2.py --raw ~/Downloads/DPD_Small --out ~/Downloads/DPD_Small_clean_v2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPORTS = REPO / "reports"

_spec = importlib.util.spec_from_file_location("build_v1", REPO / "02_build_clean.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

SIZE, JPEG_QUALITY, SEED, LABEL_COLS = v1.SIZE, v1.JPEG_QUALITY, v1.SEED, v1.LABEL_COLS

# Held-out pair rule (see V2_CHANGES.md). The result is checked against this list so a silent change fails loudly.
EXPECTED_HELDOUT = sorted(["apple_mosaic_virus", "potato_late_blight", "pear_spot",
                           "tomato_leaf_blight", "corn_yellowing", "peach_curl"])
HELDOUT_TARGET_IMAGES = 450  # cap; the rule currently yields 6 pairs / 227 images, below the cap


def select_heldout(train: pd.DataFrame, unseen: pd.DataFrame) -> list[str]:
    """Pick seen pairs that look like the unseen test: crop and disease both still in train on other pairs."""
    un_dis, un_crop = set(unseen.disease), set(unseen.crop)
    crop_n = train.crop.value_counts()
    g = train.groupby("comp").agg(n=("filename", "size"), crop=("crop", "first"), dis=("disease", "first"))

    def ok(c: str) -> bool:
        r, rest = g.loc[c], train[train.comp != c]
        return (r.dis != "healthy"
                and r.dis not in un_dis                                   # never weaken an unseen-test disease
                and (r.crop not in un_crop or crop_n[r.crop] >= 500)      # only big unseen-test crops
                and 10 <= r.n <= 100
                and (rest.disease == r.dis).sum() >= 10                   # disease still learnable elsewhere
                and rest[rest.crop == r.crop].comp.nunique() >= 2
                and (rest.crop == r.crop).sum() >= 30)                    # crop still learnable elsewhere

    rng = np.random.default_rng(SEED)
    picked, used_crop, used_dis, total = [], set(), set(), 0
    for c in rng.permutation(sorted(c for c in g.index if ok(c))):
        r = g.loc[c]
        if r.crop in used_crop or r.dis in used_dis:
            continue
        picked.append(c); used_crop.add(r.crop); used_dis.add(r.dis); total += r.n
        if total >= HELDOUT_TARGET_IMAGES:
            break
    return sorted(picked)


def resize_one(job: tuple[str, str]) -> tuple[str, list[float], list[float]]:
    src, dst = job
    with Image.open(src) as im:
        out = v1.to_rgb(im).resize((SIZE, SIZE), Image.Resampling.BICUBIC)  # same as v1: plain resize, no crop
    out.save(dst, "JPEG", quality=JPEG_QUALITY)
    px = np.asarray(out, dtype=np.float64) / 255.0
    return os.path.basename(dst), px.mean(axis=(0, 1)).tolist(), (px ** 2).mean(axis=(0, 1)).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="folder with DPD_Small_dataset/ and the 3 official CSVs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw_imgs, out = args.raw.expanduser() / "DPD_Small_dataset", args.out.expanduser()

    stats = pd.read_csv(REPORTS / "image_stats.csv")
    pairs = pd.read_csv(REPORTS / "near_dups.csv")
    assert stats.error.isna().all(), "decode errors found: fix before build"

    # ---- v1, reproduced exactly (same code path as 02_build_clean.main) ----
    m = stats.copy()
    m["exclude_reason"] = ""
    leak = pairs[(pairs.split_a == "train") & (pairs.split_b.str.startswith("test"))].filename_a
    leak = pd.concat([leak, pairs[(pairs.split_b == "train") & (pairs.split_a.str.startswith("test"))].filename_b])
    m.loc[m.filename.isin(leak), "exclude_reason"] = "near_duplicate_of_test_image"
    in_train = (m.split == "train") & (m.exclude_reason == "")
    m.loc[in_train, "split"] = v1.make_val_split(m[in_train], pairs)
    m.loc[m.exclude_reason != "", "split"] = "excluded"
    m["split_v1"] = m.split
    m["path"] = "images/" + m.filename.str.rsplit(".", n=1).str[0] + ".jpg"
    assert m.path.is_unique
    near = set(pairs.filename_a) | set(pairs.filename_b)
    m["flag_near_dup"] = m.filename.isin(near)
    m["flag_upscaled"] = m[["width", "height"]].min(axis=1) < SIZE
    m["real_format"] = m.real_format.replace({"MPO": "JPEG"})
    v1_sizes = m.split.value_counts().to_dict()
    assert v1_sizes == {"train": 7200, "test_seen": 2000, "test_unseen": 1448, "val": 795, "excluded": 5}, v1_sizes

    # ---- v2: held-out pairs -> val_unseen ----
    official_train = m[m.split.isin(["train", "val"])]
    heldout = select_heldout(official_train, m[m.split == "test_unseen"])
    assert heldout == EXPECTED_HELDOUT, f"held-out selection changed: {heldout}"
    m.loc[m.split.isin(["train", "val"]) & m.comp.isin(heldout), "split"] = "val_unseen"
    m["heldout_pair"] = m.comp.isin(heldout)

    # ---- v2: metadata (computed on the v2 train split) ----
    tr = m[m.split == "train"]
    dis_crops = tr.groupby("disease").crop.nunique()
    src_crops = tr.groupby("source").crop.nunique()
    comp_srcs = tr.groupby("comp").source.nunique()
    m["disease_n_train_crops"] = m.disease.map(dis_crops).fillna(0).astype(int)
    m["source_n_train_crops"] = m.source.map(src_crops).fillna(0).astype(int)
    m["source_in_train"] = m.source.isin(set(tr.source))
    m["comp_n_train_sources"] = m.comp.map(comp_srcs).fillna(0).astype(int)

    # ---- images (identical pixels to v1) ----
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    keep = m[m.split != "excluded"]
    jobs = [(str(raw_imgs / f), str(out / p)) for f, p in zip(keep.filename, keep.path)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(resize_one, jobs, chunksize=64))
    train_names = set(tr.path.str.removeprefix("images/"))
    ms = np.array([[r[1], r[2]] for r in res if r[0] in train_names])
    mean = ms[:, 0].mean(axis=0)
    std = np.sqrt(ms[:, 1].mean(axis=0) - mean ** 2)

    # ---- CSVs ----
    cols = ["path", "filename", *LABEL_COLS, "source", "flag_near_dup", "flag_upscaled", "heldout_pair",
            "disease_n_train_crops", "source_n_train_crops", "source_in_train", "comp_n_train_sources"]
    splits = ["train", "val", "val_unseen", "test_seen", "test_unseen"]
    for s in splits:
        m[m.split == s][cols].to_csv(out / f"{s}.csv", index=False)
    m[["path", "filename", "split", "split_v1", "exclude_reason", *LABEL_COLS, "source", "width", "height", "mode",
       "real_format", "ext_mismatch", *cols[9:]]].to_csv(out / "manifest.csv", index=False)

    def id_map(id_col: str, name_col: str) -> dict[int, str]:
        return {int(k): v for k, v in m.drop_duplicates(id_col).sort_values(id_col)[[id_col, name_col]].values}

    comps = m.drop_duplicates("comp_id").sort_values("comp_id")
    unseen_ids = sorted(m[m.split == "test_unseen"].comp_id.unique().tolist())
    heldout_ids = sorted(m[m.split == "val_unseen"].comp_id.unique().tolist())
    label_maps = {
        "num_crops": int(m.crop_id.nunique()), "num_diseases": int(m.disease_id.nunique()),
        "num_comps": int(m.comp_id.nunique()),
        "crop": id_map("crop_id", "crop"), "disease": id_map("disease_id", "disease"), "comp": id_map("comp_id", "comp"),
        "comp_to_crop_disease": {int(r.comp_id): [int(r.crop_id), int(r.disease_id)] for r in comps.itertuples()},
        "seen_comp_ids": sorted(set(comps.comp_id) - set(unseen_ids)), "unseen_comp_ids": unseen_ids,
        "heldout_comp_ids": heldout_ids,
        "train_comp_ids": sorted(tr.comp_id.unique().tolist()),
    }
    (out / "label_maps.json").write_text(json.dumps(label_maps, indent=1), encoding="utf-8")

    summary = {
        "version": "v2", "image_size": SIZE, "seed": SEED,
        "split_sizes": {s: int((m.split == s).sum()) for s in [*splits, "excluded"]},
        "split_sizes_v1": v1_sizes,
        "heldout_pairs": {str(c): int(((m.split == "val_unseen") & (m.comp == c)).sum()) for c in heldout},
        "test_seen_images_of_heldout_pairs": int(((m.split == "test_seen") & m.heldout_pair).sum()),
        "train_mean_rgb": mean.round(4).tolist(), "train_std_rgb": std.round(4).tolist(),
        "imagenet_mean_rgb": [0.485, 0.456, 0.406], "imagenet_std_rgb": [0.229, 0.224, 0.225],
        "train_counts": {k: {int(i): int(c) for i, c in tr[k].value_counts().sort_index().items()}
                         for k in ["crop_id", "disease_id", "comp_id"]},
        "excluded": m[m.split == "excluded"][["filename", "exclude_reason"]].to_dict("records"),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    for f, dst in [("dataset.py", "dataset.py"), ("metrics.py", "metrics.py"), ("train_independent.py", "train_independent.py"), ("DATASET_README.md", "README.md"), ("V2_CHANGES.md", "V2_CHANGES.md")]:
        shutil.copyfile(HERE / f, out / dst)
    (out / "__init__.py").touch()

    print("split sizes:", summary["split_sizes"])
    print("held-out pairs:", summary["heldout_pairs"])
    print("train mean/std:", summary["train_mean_rgb"], summary["train_std_rgb"])


if __name__ == "__main__":
    main()
