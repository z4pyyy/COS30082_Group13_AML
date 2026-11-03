# DPD_Small_clean_v2

Cleaned DPD-Small (COS30082 OOD plant disease project), version 2. The images are the same as v1; v2 adds a validation set for **unseen crop+disease pairs**, extra metadata, and a loader with an evaluation protocol shared by all three model tracks (independent crop CNN, independent disease CNN, multitask CNN).

What changed and why: see [`V2_CHANGES.md`](V2_CHANGES.md).
How the dataset was built, script by script: `V2_CHANGES.md` section 7. **Model developers don't need to run any of it.**
Team workflow for retraining on v2: `V2_WORKFLOW.md` in the project repo root.

---

## Contents

| File | What it is |
|---|---|
| `images/` | 11,443 images, 224×224 RGB JPEG (q95), EXIF rotation applied. **Identical to v1. Not augmented.** |
| `train.csv` | 6,995 training images |
| `val.csv` | 773 validation images, **seen** pairs (v1 `val.csv` minus the held-out pairs) |
| `val_unseen.csv` | **New.** 227 validation images from 6 seen pairs held out of training (see below) |
| `test_seen.csv` | 2,000, official, unchanged |
| `test_unseen.csv` | 1,448, official, unchanged (10 unseen pairs) |
| `manifest.csv` | every raw image: v2 split, v1 split, exclude reason, flags, metadata |
| `label_maps.json` | id ↔ name for crop (55), disease (175), comp (333); `comp_to_crop_disease`; `seen_comp_ids`, `unseen_comp_ids`, **`heldout_comp_ids`**, **`train_comp_ids`** |
| `stats.json` | split sizes (v2 and v1), held-out pairs, train mean/std, class counts in train, excluded files |
| `dataset.py` | PyTorch `Dataset`, augmentation, `get_loaders()`, `class_weights()`, `balanced_sampler()`, `comp_index()` |
| `metrics.py` | shared metrics for all tracks: Top-k, per-class accuracy, harmonic mean, and `evaluate_split()`, which writes the standard confusion-matrix and prediction files |
| `train_independent.py` | reference training script for the independent crop and disease tracks (`--task crop` / `--task disease`); implements the shared protocol below |
| `README.md`, `V2_CHANGES.md` | this file, and the change log |

### CSV columns

| Column | Meaning |
|---|---|
| `path` | image path relative to this folder |
| `filename` | original raw file name |
| `crop_id`, `disease_id`, `comp_id`, `crop`, `disease`, `comp` | official labels (`comp` = crop+disease pair) |
| `source` | source dataset (prefix of the file name, e.g. `CDDM`, `PWV2`) |
| `flag_near_dup` | the image has a near-duplicate somewhere in the dataset |
| `flag_upscaled` | short side was below 224 px and was upscaled |
| `heldout_pair` | **new.** the image's pair is one of the 6 held-out pairs (true for all of `val_unseen`, and for 60 images of `test_seen`) |
| `disease_n_train_crops` | **new.** number of crops this disease appears on in `train` |
| `source_n_train_crops` | **new.** number of crops this source contributes to `train` (1 = single-crop source) |
| `source_in_train` | **new.** the source appears in `train` (false for 4 sources that are only in `test_seen`) |
| `comp_n_train_sources` | **new.** number of sources this pair comes from in `train` (0 = pair not in train) |

---

## Quick start

The loader finds its CSVs and images relative to its own file, so it works wherever the folder is (Kaggle, Colab, local). Import it as a plain module:

```python
import glob, os, sys

# Kaggle: find the dataset folder, whatever the slug is
DATA = os.path.dirname(glob.glob("/kaggle/input/**/label_maps.json", recursive=True)[0])
# Colab / local: DATA = "/content/DPD_Small_clean_v2"   (copy it off Drive first, Drive is slow)
sys.path.insert(0, DATA)

from dataset import get_loaders, load_label_maps, comp_index

maps = load_label_maps()
loaders = get_loaders(batch_size=64, num_workers=2)      # keys: train, val, val_unseen, test_seen, test_unseen
for x, y in loaders["train"]:                            # x: (B,3,224,224) ImageNet-normalised
    y_crop, y_dis, y_comp = y["crop"], y["disease"], y["comp"]
    ...
```

- On Windows, put training code under `if __name__ == "__main__":` when `num_workers > 0`.
- `get_loaders(balance_by="comp_id")` switches train to a balanced sampler. Use **either** that **or** `class_weights()` in the loss, not both. The shared protocol below uses neither.

### Augmentation (train split only, on the fly)

`build_transforms(train=True)`:

| Step | Setting |
|---|---|
| `RandomResizedCrop` | 224, scale 0.6–1.0, ratio 0.8–1.25 |
| Horizontal and vertical flip | p = 0.5 each |
| `RandomRotation` | ±20° |
| `ColorJitter` | brightness / contrast / saturation 0.2, hue 0.02 |
| Normalise | ImageNet mean/std |
| `RandomErasing` | p = 0.25, 2–15% of the image |

Every other split: tensor + ImageNet normalisation only. Do not add your own augmentation, Mixup/CutMix or EMA in the shared protocol runs; extra tricks go in a separately named ablation run.

### Scoring crop+disease pairs (multitask, and for combining the two independent models)

```python
cd = comp_index()                                                     # (333, 2): comp -> [crop id, disease id]
score = log_softmax(crop_logits)[:, cd[:, 0]] + log_softmax(dis_logits)[:, cd[:, 1]]   # (B, 333)
pred = score.topk(5, dim=1).indices
```

Score over **all 333 pairs**, never only the seen ones.

---

## Shared protocol (all three model tracks)

Same rules for everyone, so the results can go in one table.

1. **Train on `train` only.** `test_seen` and `test_unseen` are used **once**, after training, for the final numbers. Never for early stopping, learning-rate choice or backbone choice.
2. **Use `get_loaders()` as is** (same augmentation, no sampler, no class weights).
3. **Settings:** image 224, batch 64, seed 42, up to 30 epochs, early-stopping patience 10, AdamW (weight decay 0.05), label smoothing 0.1. Learning rate per backbone: ResNet-50 3e-4, EfficientNet-B0 5e-4, ConvNeXt-Tiny 1e-4 (multitask may use its own, write it down).
   **Pretrained weights (pinned, ImageNet-1k):** timm `resnet50.tv2_in1k`, `efficientnet_b0.ra_in1k`, `convnext_tiny.fb_in1k`. These equal torchvision's default weights for ResNet-50 and ConvNeXt-Tiny. Do not use bare `convnext_tiny` in timm: its default is `in12k_ft_in1k` (pretrained on ImageNet-12k first), which would make the tracks unequal.
4. **Each epoch, evaluate `val` and `val_unseen`.** Top-1 on each, then their harmonic mean `H_val = 2·v·u / (v + u)`.
5. **Best epoch = highest `H_val`.** Patience counts epochs without a new best `H_val`. This mirrors the brief's final metric (harmonic mean of seen and unseen) instead of rewarding seen accuracy alone.
6. **Final evaluation on `test_seen` and `test_unseen`:** Top-1, Top-3, Top-5, harmonic mean, average accuracy per class, confusion matrices, and per-image top-5 predictions.
7. **Also report seen accuracy excluding held-out pairs** (`test_seen[~heldout_pair]`, 1,940 images). This number is comparable with v1 runs: v2 training never sees the 6 held-out pairs, so their 60 `test_seen` images are harder than in v1.

What "accuracy" means per track:

| Track | Label scored | Classes |
|---|---|---|
| Independent crop CNN | `y["crop"]` | 55 |
| Independent disease CNN | `y["disease"]` | 175 |
| Multitask CNN | the crop+disease pair (`y["comp"]`) from the scoring above; also report crop-head and disease-head accuracy | 333 pairs |

---

## The held-out pairs (`val_unseen`)

The official test has 10 unseen pairs: a seen crop and a seen disease that never appear together in training. v1 had nothing like that to tune on, so every model was tuned for seen accuracy. v2 moves 6 seen pairs out of train/val into `val_unseen`:

| Pair | Images | Disease still in `train` on | Crop images still in `train` |
|---|---|---|---|
| apple_mosaic_virus | 70 | 4 other crops (78 images) | 851 |
| potato_late_blight | 70 | tomato (101) | 130 |
| tomato_leaf_blight | 37 | 5 other crops (113) | 1,013 |
| pear_spot | 24 | 6 other crops (65) | 72 |
| corn_yellowing | 14 | 2 other crops (24) | 807 |
| peach_curl | 12 | 4 other crops (33) | 118 |

Selection rule (in `data_prep_v2/build_v2.py`, seed 42): not healthy; the disease is **not** one of the 10 unseen-test diseases; the crop is not a small unseen-test crop; 10–100 images; the disease keeps at least 10 training images on other crops; the crop keeps at least 2 other pairs and 30 images; at most one pair per crop and per disease.

`val_unseen` is small (227 images, 6 pairs), so expect it to move a few points between epochs. Look at the trend, not single epochs.

---

## Rules for the team

1. Tune only on `val` and `val_unseen`.
2. Same loader and augmentation for every backbone and every track.
3. Either `class_weights()` or `balance_by=`, never both (the shared protocol uses neither).
4. Do not merge labels. The official ids already merge synonyms (e.g. raw `corn_*` / `maize_*`). Similar names such as `spot` vs `bacterial_spot` are separate classes in the official labels.
5. Do not edit the CSVs. If you find a problem, report it so it goes into the next version.

## Data facts for the report

- Long tail: images per pair in `train` min 1, median 6, max 490.
- 77% of diseases appear on only one crop in `train`. Disease labels are therefore strongly tied to crop identity, which is the main reason disease accuracy collapses on unseen pairs.
- 87% of training pairs come from a single source dataset, and the source alone predicts the crop for 65% of training images. Backgrounds and camera style are a shortcut the models can use.
- 46 source datasets overall (42 in `train`); `CDDM` is 38% of train. 4 sources (`PDL`, `PDL3`, `BLD`, `BLD3`) appear only in `test_seen`.
- Every `test_unseen` crop and disease appears in `train`, so the unseen pairs can be scored.
