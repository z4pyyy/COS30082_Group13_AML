[Uploading README.md…]()
# DPD_Small_clean — ready for training

Clean version of DPD-Small (COS30082 OOD Plant Disease project). Built by `data_prep/01_audit.py` → `02_build_clean.py` → `03_verify.py` (all checks pass).

## Contents

| File | What it is |
|---|---|
| `images/` | 11,443 images, 224×224 RGB JPEG (q95), EXIF rotation applied |
| `train.csv` / `val.csv` | 7,200 / 795 — from official `train.csv` (val = 10% per comp, seed 42) |
| `test_seen.csv` | 2,000 — official, unchanged |
| `test_unseen.csv` | 1,448 — official, unchanged (10 unseen comps) |
| `manifest.csv` | All 11,448 raw images + split + flags + exclude reason |
| `label_maps.json` | id ↔ name for crop (55), disease (175), comp (333); `comp_to_crop_disease`; seen/unseen comp ids |
| `stats.json` | Class counts in train, train mean/std, excluded files |
| `dataset.py` | PyTorch `Dataset`, augmentation, `get_loaders()`, `class_weights()`, `balanced_sampler()` |

CSV columns: `path, filename, crop_id, disease_id, comp_id, crop, disease, comp, source, flag_near_dup, flag_upscaled`.
`path` is relative to this folder. `filename` is the original raw name.

## Quick start

```python
from DPD_Small_clean.dataset import get_loaders, load_label_maps

if __name__ == "__main__":            # required on Windows when num_workers > 0
    maps = load_label_maps()
    loaders = get_loaders(batch_size=64, num_workers=4)   # balance_by="comp_id" to over-sample rare classes
    for x, y in loaders["train"]:     # x: (B,3,224,224) ImageNet-normalised; y: {"crop","disease","comp"}
        ...
```

Backbones: `torchvision.models.resnet50 / efficientnet_b3 / convnext_tiny` with `weights="DEFAULT"`. Replace the classifier head.

## Rules for the team

1. **Tune only on `val`.** Do not use `test_seen` or `test_unseen` for early stopping, LR choice or backbone choice.
2. **Use the same loader and augmentation for all 3 backbones.** Then the backbone comparison is fair.
3. **Use either `class_weights()` in the loss or `balance_by=` in the loader. Do not use both.**
4. **Do not merge labels.** The official ids already merge synonyms (e.g. raw `corn_*`/`maize_*`).

## How baselines predict unseen comps

Crop and disease heads are trained on seen data only, but every unseen comp uses a seen crop and a seen disease.
Score each of the 333 comps from the two heads, then take the top-1 / top-5:

```python
# log_p_crop: (B,55), log_p_dis: (B,175)
cd = torch.tensor([maps["comp_to_crop_disease"][c] for c in range(maps["num_comps"])])  # (333,2)
comp_score = log_p_crop[:, cd[:, 0]] + log_p_dis[:, cd[:, 1]]                             # (B,333)
```

## Evaluation (brief section 3.3)

- Top-1 and Top-5 accuracy on comp: seen = `test_seen`, unseen = `test_unseen`.
- Harmonic mean = 2·seen·unseen / (seen + unseen).
- Confusion matrix on `test_seen`.
- Also report crop accuracy and disease accuracy separately (needed for the "why is crop easier" analysis).

## What cleaning changed

| Item | Result | Action |
|---|---|---|
| Missing / corrupt / empty files | 0 | — |
| Images in 2 official splits | 0 | — |
| Near-duplicates train ↔ test_seen | 5 (same photo) | **Train copy removed** (listed in `stats.json` → `excluded`) |
| Other near-duplicates | 39 pairs inside one split | Kept. Train pairs stay in the same train/val side. |
| Wrong extension (PNG/WEBP named .jpg, JPEG named .png) | 11 | Re-saved as JPEG |
| RGBA images | 28 | Alpha composited onto white |
| Short side < 224 px | 349 | Upscaled, `flag_upscaled=True` |
| Resize | median aspect 1.0 (p5 0.75, p95 1.54) | Plain resize to 224×224, no crop (keeps edge lesions) |

## Data facts for the report

- Long tail: train images per comp min 1, median 7, max 544. 40 comps have 1 train image (these are train-only, not in val).
- 46 source datasets. `CDDM` is ~38% of train. Use `source` to compare lab vs field images.
- Val covers 204 of 323 seen comps (comps with < 5 train images are train-only).
