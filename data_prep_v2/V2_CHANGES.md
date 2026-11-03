# DPD_Small_clean_v2: what changed, and why

**Short version:** same images and same official test sets as v1. v2 adds a validation set of **held-out crop+disease pairs** (`val_unseen.csv`), five metadata columns, a shared evaluation protocol, and a loader that supports all of it. The goal is to let every model be tuned and compared on what the brief actually measures: the harmonic mean of seen and unseen accuracy.

---

## 1. Original vs v1 vs v2

| | Original DPD-Small | v1 `DPD_Small_clean` | v2 `DPD_Small_clean_v2` |
|---|---|---|---|
| Images | 11,448, mixed sizes and formats (PNG/WEBP named .jpg, RGBA, EXIF-rotated) | 11,443, 224×224 RGB JPEG, EXIF applied | **same as v1** (rebuilt from raw with the v1 code; identical processing) |
| Near-duplicates of test images in train | 5 | removed | removed |
| Train | 8,000 (official) | 7,200 | **6,995** |
| Validation (seen pairs) | none | `val.csv` 795 (10% per pair, near-duplicate-aware) | `val.csv` **773** (v1 val minus held-out pairs) |
| Validation (unseen pairs) | none | none | **`val_unseen.csv` 227** (6 held-out pairs) |
| `test_seen` / `test_unseen` | 2,000 / 1,448 | unchanged | **unchanged** (verified identical to the official CSVs) |
| Metadata | labels only | + `source`, `flag_near_dup`, `flag_upscaled` | + `heldout_pair`, `disease_n_train_crops`, `source_n_train_crops`, `source_in_train`, `comp_n_train_sources` |
| Loader | none | `dataset.py` | `dataset.py` for v2 (new splits, `comp_index()`, root-relative paths for Kaggle) |
| Evaluation protocol | brief section 3.3 | README rules (tune on val only, same loader for all) | **shared protocol**: best epoch by harmonic mean of `val` and `val_unseen`; extra reporting (see README) |
| Augmentation | none | on the fly in the loader | on the fly in the loader (images on disk are never augmented) |

Where the 205 training images went (7,200 → 6,995) and the 22 validation images (795 → 773): all 227 are now in `val_unseen`.

---

## 2. Why v2 exists: what the v1 results showed

Three v1 pipelines were trained: an independent crop CNN, an independent disease CNN, and a multitask CNN (frozen ConvNeXt-Tiny). They differed in augmentation (none / strong / shared loader), validation set, backbone training (fine-tuned / frozen) and task. They still agreed on everything that matters:

| ConvNeXt-Tiny, Top-1 % | Seen | Unseen |
|---|---|---|
| Crop (independent) | 91.8 | 77.6 |
| Multitask crop head | 87.5 | 62.2 |
| Disease (independent) | 70.9 | 9.8 |
| Multitask disease head | 59.4 | 5.1 |
| Multitask crop+disease pair | 58.0 | 3.0 |

What was found in the data:

1. **Disease labels are tied to crop.** 76% of diseases appear on only one crop in the official train set (77% in v2 train). A model scores well on seen images by learning "this crop → its usual diseases". On unseen pairs, 52–60% of the independent disease model's predictions were a disease the same crop had in training (tomato early_blight → septoria / late_blight; grape black_rot → esca). The one unseen disease that transfers, powdery_mildew (59–86% across pipelines), is spread over many crops in training.
2. **Each source dataset has its own look.** 86% of training pairs come from one source, and the source alone predicts the crop for 64% of training images. Several unseen pairs come from single-crop sources (MLBD mango, SLDD sugarcane, COPD cotton, CDRD cucumber), and the model predicts a disease that source had in training.
3. **v1 had no way to tune for unseen pairs.** Every model picked its best epoch on seen accuracy only, while the brief's headline metric is the harmonic mean of seen and unseen.
4. **The three pipelines did not use the same data pipeline**, so their results could not be compared directly.

---

## 3. What v2 changes, and what each change is for

| Change | Addresses | What to expect |
|---|---|---|
| `val_unseen.csv`, 6 held-out pairs | finding 3 | Model selection can now reward unseen-pair performance without touching the test sets. It will **not** by itself fix finding 1, but it measures it during training and stops models being chosen for seen accuracy alone. |
| Best epoch by `H_val` (harmonic mean of `val` and `val_unseen` Top-1) | finding 3 | Usually an earlier or less over-fitted epoch than "best seen val". |
| One loader and one protocol for all three tracks | finding 4 | The three tracks' results can go in one table. |
| Metadata columns | findings 1 and 2 | Lets the report break accuracy down by how many crops a disease is seen on, and by single-crop sources, without re-deriving it. |
| `heldout_pair` flag on `test_seen` | side effect of the held-out pairs | 60 `test_seen` images belong to the held-out pairs, which v2 training never sees. Report seen accuracy with and without them; the "without" number (1,940 images) is comparable with v1. |

### How the held-out pairs were chosen

Rule in `data_prep_v2/build_v2.py` (seed 42):
- the pair is not `healthy`;
- its disease is **not** one of the 10 unseen-test diseases (holding one out would remove training examples the unseen test needs; e.g. `apple_black_rot` is 21 of the 36 black_rot training images, the only source for `grape_black_rot`);
- its crop is not a small unseen-test crop (only tomato and corn, with 500+ images, are allowed);
- 10–100 training images;
- the disease keeps at least 10 training images on other crops, and the crop keeps at least 2 other pairs and 30 images;
- at most one pair per crop and per disease.

Result: `apple_mosaic_virus` (70), `potato_late_blight` (70), `tomato_leaf_blight` (37), `pear_spot` (24), `corn_yellowing` (14), `peach_curl` (12). 227 images, 6 pairs.

---

## 4. What was tried and left out

| Idea | Result | Decision |
|---|---|---|
| **Background removal** (u2net via `rembg`) on every split, to remove the source/background shortcut | Works on lab photos of single leaves (CDDM, PDL, BLD, CDC). Fails on field photos: erased most of the image for PWV2, SLDD, COPD, PDN, FP and PD samples, kept only the lesion for BLSD banana, and kept the background instead of the leaf for MLBD mango. Five of the failing sources (PWV2, SLDD, MLBD, COPD, PDN) supply 47% of `test_unseen`. Segmenting only where it works would make "segmented or not" a new source signal. | **Not included.** A better segmenter or a training-only background-swap augmentation could be a v3. |
| Merging similar disease names (`spot` vs `bacterial_spot`, `mosaic` vs `mosaic_virus`) | Would change the official ids and break comparison with the brief. | **Not done.** The names are flagged in the README for the report. |
| Rebalancing or resampling the data on disk | Would change the dataset's distribution for every track. | **Not done.** `balanced_sampler()` and `class_weights()` remain available as opt-in loader options for ablations. |

---

## 5. Caveats

- **The v2 loader is a fresh implementation.** v1's `dataset.py` was not available when v2 was built, so v2's `dataset.py` was written to the same interface (`get_loaders`, `load_label_maps`, `DPDDataset`, `build_transforms`, `class_weights`, `balanced_sampler`). The augmentation settings are listed in the README. They may differ slightly from v1's, so compare v1 and v2 runs on trends, not on fractions of a point.
- **`val_unseen` is small** (227 images, 6 pairs, two of them over 60% of it). Treat single-epoch changes of a few points as noise.
- **The held-out pairs are not the unseen-test pairs.** By design they avoid the unseen-test diseases, so `val_unseen` is a proxy for unseen behaviour, not a copy of the test.
- **Seen test accuracy will drop slightly vs v1** because of the 60 held-out-pair images (3% of `test_seen`), and because training has 205 fewer images. Use the "excluding held-out pairs" number to compare with v1.
- **Pretrained weights were not the same across v1 tracks.** The v1 independent notebooks used timm defaults (`convnext_tiny` = `in12k_ft_in1k`, pretrained on ImageNet-12k first; `resnet50` = `a1_in1k`), while multitask used torchvision ImageNet-1k weights. v2's shared protocol pins `convnext_tiny.fb_in1k` and `resnet50.tv2_in1k` (the same weights torchvision uses) and `efficientnet_b0.ra_in1k`. v1 independent ConvNeXt numbers may therefore be a little higher than a v2 run would give for that reason alone.

---

## 6. Files added for the model tracks

| File | Purpose |
|---|---|
| `metrics.py` | the same metric code for everyone: Top-1/3/5, per-class Top-1, harmonic mean, and `evaluate_split()`, which writes `<Model>_<split>_confusion_matrix.csv/.png` and `<Model>_predictions_<split>.csv` |
| `train_independent.py` | reference training script for the independent crop and disease tracks (`--task crop` or `--task disease`), implementing the shared protocol and writing the standard output folder |

---

## 7. Data preparation pipeline (what each script does)

**Model developers do not run any of this.** The Kaggle dataset is already built and verified. This section is here so everyone can describe the data preparation correctly in the report, and so the dataset maintainer can rebuild it.

```
raw DPD_Small (11,448 images + 3 official CSVs)
   │
   ├─ 01_audit.py ───────► reports/  image_stats.csv, near_dups.csv, audit_summary.txt
   │
   ├─ 02_build_clean.py ─► DPD_Small_clean/        (v1)   ── 03_verify.py checks it
   │      (uses reports/)
   │
   └─ data_prep_v2/build_v2.py ─► DPD_Small_clean_v2/ (v2) ── data_prep_v2/verify_v2.py checks it ──► Kaggle
          (uses reports/ and the functions in 02_build_clean.py)
```

| Script | Input | What it does | Output | Key results |
|---|---|---|---|---|
| `01_audit.py` (v1) | raw images + 3 official CSVs | Fully decodes every image (catches truncated files); records width, height, colour mode and real file format; computes a perceptual hash (pHash) per image; lists near-duplicate pairs (pHash distance ≤ 4) and flags wrong file extensions. Changes nothing. | `reports/image_stats.csv`, `reports/near_dups.csv`, `reports/audit_summary.txt` | 11,448 images, 0 decode errors, 11 wrong extensions, 28 RGBA, 349 with a short side under 224 px, 44 near-duplicate pairs (5 between train and `test_seen`, 1 with different labels) |
| `02_build_clean.py` (v1) | raw images + `reports/` | Removes the 5 training copies of `test_seen` near-duplicates. Splits 10% of each training pair into `val` (pairs under 5 images stay in train; near-duplicate groups are kept on the same side, so no near-copy leaks into val). For every image: applies the EXIF rotation, composites transparency onto white, converts to RGB, resizes to 224×224 (plain bicubic resize, no crop, so edge lesions are kept) and saves as JPEG quality 95. Writes the CSVs, `manifest.csv`, `label_maps.json`, `stats.json` (class counts, train mean/std). | `DPD_Small_clean/` | train 7,200 / val 795 / test_seen 2,000 / test_unseen 1,448; 5 excluded |
| `03_verify.py` (v1) | `DPD_Small_clean/` | Checks every file exists, labels match `label_maps.json`, no image is in two splits, no unseen pair is in train; runs the loaders; saves an augmentation preview (one image plus 7 random training augmentations, a visual check of the loader, not saved data); does a GPU forward pass. | console report, `reports/augmentation_preview.png` | all checks passed |
| `data_prep_v2/build_v2.py` (v2) | raw images + `reports/` | Re-runs v1's exclusion and val split with v1's own functions, and asserts the v1 split sizes come out exactly. Then selects the 6 held-out pairs (rule in section 3) and moves their train/val images to `val_unseen`; adds the 5 metadata columns; writes the images with the same processing as v1; writes CSVs, `manifest.csv` (with each image's v1 split too), `label_maps.json` (plus `heldout_comp_ids`, `train_comp_ids`), `stats.json`; copies in `dataset.py`, `metrics.py`, `train_independent.py`, `README.md`, `V2_CHANGES.md`. | `DPD_Small_clean_v2/` | train 6,995 / val 773 / val_unseen 227 / test_seen 2,000 / test_unseen 1,448; 5 excluded |
| `data_prep_v2/verify_v2.py` (v2) | `DPD_Small_clean_v2/` + official CSVs | 8 checks: test sets identical to the official CSVs; train + val + val_unseen = official train minus the 5 exclusions; no image in two splits; held-out pairs absent from train/val with their crop and disease still in train; every unseen-test crop and disease in train; label maps match every CSV; all 11,443 images present and 224×224 RGB; the loader, `class_weights()` (v1 and v2 call styles) and `comp_index()` work. | console report | all 8 passed |

What data preparation does **not** do: it never augments images on disk (augmentation happens in `dataset.py` during training), never changes labels, and never touches the official test sets.

Note for the maintainer: the v1 scripts were written for a `data_prep/` folder layout (`data_prep/reports/`, `DPD_Small/` next to it). In this repo they sit at the root with `reports/` beside them, so restore that layout before re-running them. The v2 scripts take explicit `--raw` / `--out` / `--data` paths and work with the repo as it is.

---

## 8. Reproduce

From the repo root, with the raw dataset (`DPD_Small_dataset/` and the 3 official CSVs) in `~/Downloads/DPD_Small`:

```
python data_prep_v2/build_v2.py  --raw ~/Downloads/DPD_Small --out ~/Downloads/DPD_Small_clean_v2
python data_prep_v2/verify_v2.py --raw ~/Downloads/DPD_Small --data ~/Downloads/DPD_Small_clean_v2
```

Build needs `pandas numpy pillow`; verify also needs `torch torchvision`. The build reuses v1's `02_build_clean.py` and the v1 audit outputs in `reports/`, and asserts that the v1 split sizes and the held-out selection come out exactly as documented.
