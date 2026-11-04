# Independent crop CNN on DPD_Small_clean_v2: crop_independent_v2

**Author:** TODO (your name)  ·  **Date:** 2026-11-04  ·  **Track:** independent crop CNN
**Run type:** main protocol run

## 1. Summary

Trained the independent crop classifier (55 classes) with three backbones (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny) on DPD_Small_clean_v2 with the shared v2 protocol, using the dataset's `train_independent.py` unchanged. The best model by harmonic mean is **ConvNeXt-Tiny: seen 88.90 / unseen 70.23 / H 78.47 (Top-1, %)**; EfficientNet-B0 (H 76.97) and ResNet-50 (H 75.01) are close behind.

The main takeaway: seen crop accuracy is high (88 to 89%), but unseen is much lower (65 to 70%) and is flattered by tomato (33% of `test_unseen`, 82 to 85% correct). Cherry is 0% for all three models, and per-class seen accuracy is only about 60% against 88 to 89% overall (long tail). One handoff check failed: EfficientNet-B0's clean train Top-1 is exactly 100% (see section 10).

## 2. Environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4, 15,360 MiB |
| Python / torch / torchvision / timm | 3.13.16 / 2.11.0+cu130 / not recorded / 1.0.30 |
| Dataset | DPD_Small_clean_v2, Kaggle version TODO (the notebook used the placeholder value "2"; confirm with the group) |
| Repo commit | TODO (the notebook used the placeholder "abc1234"; use the real commit from the `[v2 dataset] READY` message) |
| Notebook | `crop_independent_cnns_v2.ipynb` (committed with outputs) |

## 3. Exactly how to reproduce

Colab, T4 runtime, dataset zip (`archive.zip`, the Kaggle download of DPD_Small_clean_v2) stored in Google Drive at `MyDrive/archive.zip`:

```
# 1. Mount Drive, copy archive.zip to local disk, unzip it, and keep the folder that contains
#    train_independent.py and label_maps.json as /content/DPD_Small_clean_v2

# 2. Install timm
pip -q install timm

# 3. Sanity check: GPU present, split sizes 6995 / 773 / 227 / 2000 / 1448 (asserted in the notebook)

# 4. Smoke test (1 epoch, 3 batches, ResNet-50 only), then delete it
python /content/DPD_Small_clean_v2/train_independent.py --task crop --out /content/smoke --models ResNet50 --epochs 1 --limit-batches 3 --workers 2
rm -rf /content/smoke

# 5. Main run (all 3 backbones, script unchanged), written to LOCAL disk
python /content/DPD_Small_clean_v2/train_independent.py --task crop --workers 2 --out /content/crop_independent_v2

# 6. Copy the results to Drive
mkdir -p /content/drive/MyDrive/DPD_runs/v2/crop_independent_v2
cp -r /content/crop_independent_v2/. /content/drive/MyDrive/DPD_runs/v2/crop_independent_v2/
```

The run writes to local disk first because writing `--out` straight to the Drive folder failed (see section 5, attempt 1). The notebook's later cells (file count, results table, SHA-256) read the Drive copy.

## 4. Pipeline

- **Data:** the dataset loader used by `train_independent.py`, splits train 6,995 / val 773 / val_unseen 227 / test_seen 2,000 / test_unseen 1,448 (checked in the notebook; the log's first line shows the same sizes). Nothing different.
- **Augmentation:** the dataset loader's train augmentation, nothing added.
- **Sampler / class weights:** none.
- **Model(s):** timm `resnet50.tv2_in1k`, `efficientnet_b0.ra_in1k`, `convnext_tiny.fb_in1k` (pinned ImageNet-1k weights), classifier head replaced for 55 classes.
- **Loss:** CrossEntropy with label smoothing 0.1.
- **Optimiser and schedule:** AdamW, weight decay 0.05, OneCycleLR (pct_start 0.1), mixed precision (amp). Learning rates: ResNet-50 3e-4, EfficientNet-B0 5e-4, ConvNeXt-Tiny 1e-4.
- **Epochs / patience / batch / seed:** 30 / 10 / 64 / 42.
- **Model selection:** best epoch = highest H(val Top-1, val_unseen Top-1).
- **Changes from the shared protocol:** none to the training protocol. Colab-specific differences only: `--workers 2` (the handoff's Colab command) instead of 4, and output written to local disk and copied to Drive afterwards.

## 5. Process log

| # | Date | What was run | Result | Decision |
|---|---|---|---|---|
| 1 | 2026-11-04 | Setup: mount Drive, get dataset into `/content/DPD_Small_clean_v2` | Several failed attempts: placeholder values left in the config cell, dataset given as a zip rather than a folder, and a re-run that skipped unzipping because a wrong folder from an earlier attempt already existed | Rewrote the setup cell to always re-extract from the zip and to pick the folder that contains `train_independent.py`. Final listing shows all expected files including `train_independent.py` and `val_unseen.csv` |
| 2 | 2026-11-04 | Sanity check: GPU and split sizes | Tesla T4, `cuda True`, sizes exactly 6995 / 773 / 227 / 2000 / 1448 | OK |
| 3 | 2026-11-04 | Smoke test, ResNet-50, 1 epoch, 3 batches | Ran end to end, all output files written (numbers meaningless, 12 s/epoch) | Deleted the smoke folder, went on |
| 4 | 2026-11-04 | Main run attempt 1, `--out` on the Drive folder | ResNet-50 trained (early stop at epoch 19, best epoch 9, H 93.94), then crashed with `FileNotFoundError` when reloading `ResNet50_best.pt` from the Drive path. Training was lost | Stopped. Cause: reading a just-written checkpoint back from the Drive-mounted folder is unreliable. Decided to train on local disk and copy to Drive afterwards. The late-arriving checkpoint from this attempt later showed up in Drive as `ResNet50_best (1).pt` |
| 5 | 2026-11-04 | Main run attempt 2, `--out /content/crop_independent_v2`, then copy to Drive | Completed: ResNet-50 (19 epochs), EfficientNet-B0 (30), ConvNeXt-Tiny (19). The ResNet-50 log matches attempt 1 epoch for epoch (same losses and accuracies, only the seconds differ) | This is the main run. The ResNet-50 result reproduces exactly with seed 42 |
| 6 | 2026-11-04 | Output checks | 47 files as expected, `config.json` shows `DPD_Small_clean_v2` and a Tesla T4, `summary.csv` has 3 rows, all 3 checkpoints reload. The sanity check flagged one problem: EfficientNet-B0 `train_clean_top1` is 100.0% | Not pushed yet. See section 10 |

## 6. Training behaviour

Mean seconds per epoch is from each backbone's `_history.csv`; total time is epochs run times that mean (training epochs only, approximate). Curves: [ResNet50](ResNet50_curves.png), [EfficientNet-B0](EfficientNet-B0_curves.png), [ConvNeXt-Tiny](ConvNeXt-Tiny_curves.png).

| Model | Best epoch | Epochs run | Time / epoch | Total time | Clean train Top-1 | Val Top-1 | Val-unseen Top-1 |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 9 | 19 | 69 s | about 22 min | 99.37 | 94.05 | 93.83 |
| EfficientNet-B0 | 25 | 30 | 67 s | about 34 min | 100.00 | 96.12 | 88.55 |
| ConvNeXt-Tiny | 9 | 19 | 72 s | about 23 min | 98.40 | 95.73 | 88.55 |

- **Early stopping:** ResNet-50 and ConvNeXt-Tiny peaked at epoch 9 and stopped at epoch 19 (patience 10). EfficientNet-B0 improved late (best epoch 25) and ran all 30 epochs.
- **Train vs val:** augmented train accuracy reaches 99.9% or more for all three (EfficientNet-B0 99.99%) while val sits at 95 to 96%, so the models fit the training set almost perfectly. Train loss flattens at about 0.73, which is near the floor for label smoothing 0.1.
- **val_unseen is noisy:** it has only 227 images and swings by several points between epochs (ResNet-50 ranges from 80.2 to 93.8 after epoch 9). ResNet-50's best epoch (9, val_unseen 93.83) is a spike: its neighbours are 86.3 and 89.4, and it was never matched later. The best epoch is therefore partly luck.
- **val_unseen vs test_unseen:** val_unseen is 88.6 to 93.8% but test_unseen is only 65.3 to 70.2%. The 6 held-out validation pairs are much easier than the 10 unseen test pairs, so selecting on val_unseen does not predict unseen test accuracy well.

## 7. Results (test sets, used once)

Copied from `summary.csv`. All values in %.

| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | Seen excl. held-out Top-1 | Seen per-class | Unseen per-class |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 88.05 | 65.33 | 75.01 | 95.45 | 83.22 | 88.92 | 88.20 | 59.57 | 61.51 |
| EfficientNet-B0 | 89.35 | 67.61 | 76.97 | 94.75 | 81.70 | 87.74 | 89.43 | 62.48 | 63.49 |
| ConvNeXt-Tiny | 88.90 | 70.23 | **78.47** | 95.35 | 83.98 | **89.30** | 89.02 | 59.87 | 64.69 |

**Compared with v1** (ConvNeXt-Tiny, the only v1 crop numbers I have: from `V2_CHANGES.md` and the handoff; "seen" for v2 is seen excl. held-out):

| | Seen Top-1 | Seen per-class | Unseen Top-1 |
|---|---|---|---|
| v1 ConvNeXt-Tiny | 91.8 | 67.5 | 77.6 |
| v2 ConvNeXt-Tiny | 89.02 | 59.87 | 70.23 |

v2 is lower on every number. This is expected rather than a regression: v1 picked its best epoch on `test_seen` (a leak), used no augmentation, and used timm's default ImageNet-12k ConvNeXt weights. v2 changed all of these at once, so the drop cannot be attributed to any single change. v1 numbers for ResNet-50 and EfficientNet-B0 are TODO (not in the material used for this README).

## 8. Error analysis

From each backbone's `_predictions_unseen.csv`. Per-class Top-1 on `test_unseen` (1,448 images; the 8 crops below have at least 10 images each and cover 1,413 of them). Confusion matrices: [ResNet50](ResNet50_unseen_confusion_matrix.png), [EfficientNet-B0](EfficientNet-B0_unseen_confusion_matrix.png), [ConvNeXt-Tiny](ConvNeXt-Tiny_unseen_confusion_matrix.png).

| Crop | Images (share of test_unseen) | ResNet-50 | EfficientNet-B0 | ConvNeXt-Tiny |
|---|---|---|---|---|
| tomato | 472 (32.6%) | 82.8 | 84.5 | 81.8 |
| grape | 271 (18.7%) | 57.6 | 59.8 | 64.6 |
| pepper_bell | 220 (15.2%) | 49.1 | 54.1 | 74.1 |
| cotton | 125 (8.6%) | 61.6 | 57.6 | 68.0 |
| mango | 95 (6.6%) | 78.9 | 93.7 | 82.1 |
| sugarcane | 89 (6.1%) | 88.8 | 85.4 | 75.3 |
| cherry | 72 (5.0%) | **0.0** | **0.0** | **0.0** |
| cucumber | 69 (4.8%) | 37.7 | 42.0 | 44.9 |

- **Tomato flatters the unseen number.** It is a third of `test_unseen` and scores 82 to 85%, while the crops with few training images score far lower. This matches the v1 pattern described in the handoff.
- **Cherry is 0% for all three backbones** (72 images). It was 33% in v1 according to the handoff, so it is worse here. Cucumber (38 to 45%) is similar to v1 (42%). Pepper_bell (49 to 74%) is similar to or better than v1 (62%) depending on the backbone.
- **ConvNeXt-Tiny wins on unseen mainly through pepper_bell** (74.1% against 49.1% for ResNet-50 and 54.1% for EfficientNet-B0), plus grape and cotton. It is worse than the others on sugarcane.
- **Per-class vs overall (long tail):** seen is 88 to 89% overall but only 60 to 62% averaged per class. Unseen is 65 to 70% overall and 62 to 65% per class, a smaller gap because the largest class (tomato) is the easiest.
- TODO: what cherry, cucumber and pepper_bell are confused with is not printed in the notebook. Open the `unseen` confusion matrices above and add the main confusions here. A likely reason (to be checked, not shown by these outputs) is that these crops have few training images, all of other diseases.

## 9. Checkpoints

Not in git (`.gitignore` excludes `*.pt`; GitHub rejects files over 100 MB). Sent to the team WhatsApp group as documents, right after the "done" message.

| File | Size | SHA-256 | Epoch | Val H | Sent to WhatsApp group (date) |
|---|---|---|---|---|---|
| ResNet50_best.pt | 91 MB | `15766f26b985688f9aaf9cc051bbf0d0e0a23d10ae9244699cadcca5076a5ba7` | 9 | 93.94 | TODO |
| EfficientNet-B0_best.pt | 16 MB | `ef7ba59fcf870efc9581d8fa3b9a47c3172f07a11f7354c864ebbc0b8b7ccd4a` | 25 | 92.18 | TODO |
| ConvNeXt-Tiny_best.pt | 107 MB | `2980a866301ced2fab3d3f9df86643434e09e735a68538b23d64d009cf0daa44` | 9 | 92.00 | TODO |

How to load:

```python
import torch, timm
ck = torch.load("ConvNeXt-Tiny_best.pt", map_location="cpu")
model = timm.create_model(ck["timm_id"], pretrained=False, num_classes=ck["num_classes"])
model.load_state_dict(ck["state_dict"])
```

All three checkpoints were reloaded this way in the notebook and loaded without errors.

## 10. Known issues and open questions

- **EfficientNet-B0 clean train Top-1 is exactly 100.00%.** The handoff checklist says this should be below 100% to show augmentation is on, and the notebook's sanity check flagged it. Its augmented train accuracy was already 99.15% at epoch 7 and 99.99% by epoch 20, so the augmentation does not seem to slow memorisation much for this model. This run alone cannot show whether augmentation was applied. Check the train transform in the dataset's `dataset.py`. Per `V2_WORKFLOW.md`, post the summary table in the group before pushing.
- **Placeholder values in the notebook.** The committed notebook's STARTING message shows the placeholders `Kaggle v2, commit abc1234`. The real Kaggle version and commit are TODO (see section 2).
- **Duplicate checkpoint in Drive.** `ResNet50_best (1).pt` is left over from the failed attempt 1 and has the same SHA-256 as `ResNet50_best.pt`. Do not send or commit it; delete it from the Drive folder. It is not counted in the 47-file check because `.pt` files are excluded.
- **val_unseen is a weak model-selection signal** (small, noisy, and much easier than test_unseen; see section 6). ResNet-50's best epoch looks like a lucky spike.
- **Cherry is 0% on unseen for every backbone.** Worth investigating (section 8).
- **v1 comparison is incomplete** (ConvNeXt-Tiny only), and torchvision's version was not recorded.
- Things to try next, in a separately named folder as the handoff requires: class weights or a sampler for the long tail, stronger augmentation, and a different learning rate. Tune on `val` / `val_unseen` only.
