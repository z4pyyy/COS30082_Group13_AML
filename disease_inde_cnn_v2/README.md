# Independent disease CNN on DPD_Small_clean_v2: disease_independent_v2

**Author:** TODO  ·  **Date:** TODO  ·  **Track:** independent disease CNN
**Run type:** main protocol run

> Items marked **TODO** are values that were not in the run logs (author, dates, versions, commit, WhatsApp dates) or results of `disease_v2_extra_analysis.ipynb` that still have to be pasted in. Everything else comes from the executed notebook `disease_independent_cnns_v2.ipynb`.

## 1. Summary

Three independent disease classifiers (175 classes: ResNet-50, EfficientNet-B0, ConvNeXt-Tiny) were trained with the shared v2 protocol, using the dataset's `train_independent.py` unchanged. The best model by harmonic mean is **ConvNeXt-Tiny: seen Top-1 64.30, unseen Top-1 14.09, H Top-1 23.11** (H Top-5 47.74). All three backbones are equal on seen pairs (64.3-65.1%); ConvNeXt wins only on unseen pairs (14.1% against 8.6-8.9%). Take-away: independent disease CNNs do not transfer to unseen crop-disease pairs (Top-1 9-14%) even though every unseen crop and disease occurs in training, which fits the data: 76.6% of diseases appear on a single crop in `train`.

## 2. Environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4 |
| Python / torch / torchvision / timm | Python 3.13 / **TODO** (run `import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__)`) |
| Dataset | DPD_Small_clean_v2, Drive copy (not a Kaggle version): the CSVs and scripts come from the v2 folder and the images were completed from the v1 folder, see section 5 |
| Repo commit | TODO |
| Notebook | `disease_independent_cnns_v2.ipynb` (with outputs); extra analysis in `disease_v2_extra_analysis.ipynb` |

## 3. Exactly how to reproduce

```
# 1. Colab: Runtime -> GPU (T4). Mount Drive and build the local dataset folder /content/DPD_Small_clean_v2
#    (CSVs + scripts from DPD_Small_clean_v2, images from DPD_Small_clean/images, then the v2 images on top);
#    check: 11,443 images and 0 missing paths in the five CSVs.

pip -q install timm

# 2. optional smoke test (about 2 minutes)
python /content/DPD_Small_clean_v2/train_independent.py --task disease \
    --out /content/drive/MyDrive/DPD_runs/v2/smoke --models ResNet50 --epochs 1 --limit-batches 3

# 3. main run (all three backbones, 78 min of training on the T4)
python /content/DPD_Small_clean_v2/train_independent.py --task disease --workers 2 \
    --out /content/drive/MyDrive/DPD_runs/v2/disease_independent_v2

# 4. extra analysis (CPU is enough): run disease_v2_extra_analysis.ipynb top to bottom
```

## 4. Pipeline

- **Data:** `get_loaders()` from the dataset, splits train 6,995 / val 773 / val_unseen 227 / test_seen 2,000 / test_unseen 1,448 (first line of the log). Nothing different. The local image folder was assembled as described in section 5.
- **Augmentation:** the dataset loader's train augmentation. Nothing added.
- **Sampler / class weights:** none.
- **Model(s):** timm `resnet50.tv2_in1k`, `efficientnet_b0.ra_in1k`, `convnext_tiny.fb_in1k` (ImageNet-1k), classifier head replaced by a 175-class linear layer.
- **Loss:** CrossEntropy, label smoothing 0.1.
- **Optimiser and schedule:** AdamW, weight decay 0.05, OneCycle. Learning rates 3e-4 (ResNet-50), 5e-4 (EfficientNet-B0), 1e-4 (ConvNeXt-Tiny).
- **Epochs / patience / batch / seed:** up to 30 / 10 / 64 / 42.
- **Model selection:** best epoch = highest H(val Top-1, val_unseen Top-1). Nothing different.
- **Changes from the shared protocol:** none to the script, loader, augmentation, learning rates, epochs, patience or selection rule. Run on Colab with `--workers 2` and with the image folder assembled from v1 + v2 (section 5).

## 5. Process log

Dates were not recorded in the logs: **TODO** fill in the Date column.

| # | Date | What was run | Result | Decision |
|---|---|---|---|---|
| 1 | TODO | Kaggle notebook with the team dataset attached, smoke test and main run | Split sizes were correct, but Internet was off and there was no GPU (`device: cpu`): the account could not be phone-verified (Kaggle: "Too many requests to StartPhoneVerification"). Stopped at the pretrained-weights download (`Temporary failure in name resolution`). No results. | Abandon Kaggle, use Colab |
| 2 | TODO | Colab: local copy of the Drive folder `DPD_Small_clean_v2`, smoke test | Crash `FileNotFoundError: .../images/FN_rose_downy_mildew_402.jpg`. The Drive v2 folder held the CSVs and scripts, but `images/` had 6 files (later 3,128) instead of 11,443. | Complete the images from the v1 folder (identical images according to the dataset README) |
| 3 | TODO | Overlap check v1 vs v2 Drive images | v1: 11,443 files, v2: 3,128 files, all of them also in v1. None of the first 200 overlapping files differed by MD5. **21 overlapping files had different byte sizes** (the MD5 check did not cover them). | Copy v1 images, then the v2 images on top (so the v2 versions were used for the 3,128). Analysis of the 21 files: see section 10 |
| 4 | TODO | Local image folder rebuilt, check of the five CSVs | 11,443 images, 0 missing paths in train, val, val_unseen, test_seen, test_unseen | Continue |
| 5 | TODO | Smoke test (ResNet-50, 1 epoch, 3 batches) | Ran on the GPU, the numbers are near chance as expected (Seen Top-1 3.12). It also printed test metrics, which were not used for any decision. | Start the main run |
| 6 | TODO | Main run, all three backbones (`train_independent.py --task disease`) | Finished with all required files. ConvNeXt-Tiny best (H Top-1 23.11) | Keep as the main result |
| 7 | TODO | Extra analysis notebook | **TODO** (results go into sections 8 and 10) | |

The main run was executed once (seed 42). Nothing was tuned on `test_seen` or `test_unseen`. The Hugging Face "unauthenticated request" warning is harmless.

## 6. Training behaviour

Per-epoch values are in `<Model>_history.csv`; curves in `<Model>_curves.png` and `val_vs_val_unseen_all.png`.

| Model | Best epoch | Epochs run | Time / epoch | Total time | Clean train Top-1 | Val Top-1 | Val-unseen Top-1 |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 15 | 25 | 67.6 s | 28.1 min | 98.66 | 82.41 | 15.42 |
| EfficientNet-B0 | 15 | 25 | 64.5 s | 26.9 min | 99.36 | 80.60 | 14.10 |
| ConvNeXt-Tiny | 10 | 20 | 69.7 s | 23.2 min | 91.07 | 81.37 | 16.30 |

(Total 78.3 min of training. Clean train, val and val-unseen are for the saved best epoch.)

- **Train vs val gap:** all three overfit. Train accuracy on augmented images reaches 99.7-99.8% (ResNet-50, EfficientNet-B0) and 98.6% (ConvNeXt-Tiny) by the end, while `val` stays at 81-83%. The training loss levels off at 0.87-0.89, about the floor for label smoothing 0.1 with 175 classes.
- **How `val_unseen` behaved:** it has 227 images, so one image is 0.44 points, and it swings by 3-5 points between epochs (ResNet-50 over epochs 6-15: 10.6, 8.4, 10.1, 10.6, 8.4, 7.9, 11.9, 13.7, 12.8, 15.4). It starts near 0% and rises to roughly 10-16% from about epoch 6-10, so the chosen epoch is partly chance.
- **ConvNeXt-Tiny** learns more slowly (11.7% augmented train accuracy at epoch 1 against about 17%), but has the best `H_val` (27.16 at epoch 10) and the smallest train-val gap.
- **Early stopping** (patience 10 on `H_val`): ResNet-50 and EfficientNet-B0 stopped at epoch 25 (best 15, H_val 25.98 and 24.00), ConvNeXt-Tiny at epoch 20 (best 10).

## 7. Results (test sets, used once)

From `summary.csv` / `results.json`. All values in %.

| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | Seen excl. held-out Top-1 | Seen per-class | Unseen per-class |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 65.05 | 8.63 | 15.24 | 83.70 | 27.97 | 41.93 | 66.91 | 45.08 | 15.02 |
| EfficientNet-B0 | 64.85 | 8.91 | 15.67 | 83.75 | 26.66 | 40.44 | 66.70 | 43.55 | 13.82 |
| **ConvNeXt-Tiny** | 64.30 | **14.09** | **23.11** | 84.35 | **33.29** | **47.74** | 66.13 | 35.10 | **19.49** |

Top-3: seen / unseen / H = ResNet-50 79.00 / 21.13 / 33.35; EfficientNet-B0 79.85 / 19.54 / 31.40; ConvNeXt-Tiny 79.80 / 25.07 / 38.15.

- Seen accuracy is the same for the three backbones (about 1 point apart). The ranking comes from unseen accuracy, which rests on only 10 unseen pairs and one seed.
- Per-class accuracy is far below overall accuracy (35-45% against 64-65% on seen): frequent diseases are recognised, rare ones are not (pairs have 1-490 training images, median 6).
- Validation overstates the test accuracy: `val` is 80.6-82.4% against 64.3-65.1% on `test_seen` (about 17 points). Four source datasets appear only in `test_seen`; **TODO:** quantify this with the "seen by `source_in_train`" table from the extra analysis notebook (cell 3).
- `val_unseen` is optimistic because it picks the epoch: ResNet-50 falls from 15.4 to 8.6 on `test_unseen` and EfficientNet-B0 from 14.1 to 8.9, while ConvNeXt-Tiny holds (16.3 to 14.1).
- The 60 held-out-pair images in `test_seen` are harder: excluding them raises seen Top-1 by 1.1-1.9 points.

**Compared with v1** (use "seen excl. held-out" for seen): **TODO** (take the v1 numbers from `disease_inde_cnn/disease_independent/`; note that v1 used its own validation split and different pretrained weights for `convnext_tiny`).

## 8. Error analysis

Data fact from `train.csv`: **76.6% of diseases appear on exactly one crop** (median 1, mean 1.81, max 37), so the crop is a shortcut for the disease label.

Unseen Top-1 of ConvNeXt-Tiny by the number of crops the true disease has in train (`ConvNeXt-Tiny_unseen_by_n_train_crops.csv`):

| Crops in train | 2 | 3 | 5 | 8 | 9 | 11 | 12 |
|---|---|---|---|---|---|---|---|
| Top-1 (%) | 1.5 | 4.8 | 25.9 | 0.0 | 0.0 | 52.2 | 30.3 |
| Images | 544 | 125 | 491 | 35 | 95 | 69 | 89 |

Diseases spread over more crops tend to transfer better, but the trend is not clean (0% at 8 and 9 crops). The unseen set has 10 pairs, so each column holds one to three diseases: an illustration, not a statistic.

**TODO** (from `disease_v2_extra_analysis.ipynb`, cell 2):
- share of wrong unseen predictions that are a disease already seen on the true crop in training: ResNet-50 __%, EfficientNet-B0 __%, ConvNeXt-Tiny __% (the v1 analysis found 52-60%);
- most frequent wrong predictions (true crop / true disease to predicted disease) and which unseen diseases never transfer (`*_unseen_by_pair.csv`, `*_unseen_wrong_predictions.csv`);
- the same crop-count table for ResNet-50 and EfficientNet-B0.

**TODO:** 2-3 observations from the confusion matrices (`<Model>_seen_confusion_matrix.png`, `<Model>_unseen_confusion_matrix.png`): which diseases are confused on seen, and where the unseen pairs go.

## 9. Checkpoints

Not in git (`.gitignore` excludes `*.pt`; GitHub rejects files over 100 MB). Sent to the team WhatsApp group as documents, right after the "done" message.

| File | Size | SHA-256 | Epoch | Val H | Sent to WhatsApp group (date) |
|---|---|---|---|---|---|
| ResNet50_best.pt | 95.8 MB | `78521ad49b18d945314aeeac73e41a2490fa387af13286d7d9c79608fa9475e9` | 15 | 25.98 | TODO |
| EfficientNet-B0_best.pt | 17.2 MB | `fd1b94ae723bd5816687ca13fe4b27165fac976f6d82df93fe892d641f4e33f9` | 15 | 24.00 | TODO |
| ConvNeXt-Tiny_best.pt | 111.9 MB | `c18c6bbd8cb46afc4d78d5b3ad4a949024c21c808b92f4775b6898b8dd63d808` | 10 | 27.16 | TODO |

How to load:

```python
import torch, timm
ck = torch.load("ConvNeXt-Tiny_best.pt", map_location="cpu")
model = timm.create_model(ck["timm_id"], pretrained=False, num_classes=ck["num_classes"])
model.load_state_dict(ck["state_dict"])
```

## 10. Known issues and open questions

- **Wrong figure in the main notebook:** section 6 of `disease_independent_cnns_v2.ipynb` printed "0.0%" for the share of wrong unseen predictions that are a disease seen on the true crop (and two empty tables). This is a bug in my analysis cell (prediction labels were converted to numbers, which fails for class names), not a result. The corrected analysis is in `disease_v2_extra_analysis.ipynb` (cell 2); its numbers belong in section 8.
- **Images assembled from two Drive folders:** 21 overlapping files differ in byte size between the v1 and v2 Drive copies and the v2 versions were used in the run. **TODO:** result of cell 4 of the extra analysis notebook (loadable? 224x224? `mean_pixel_diff`? used in a split?). If any file is corrupt or not 224x224, the affected split(s) must be re-checked and the run repeated.
- **Val vs test gap (about 17 points):** the explanation by sources that appear only in `test_seen` is not yet measured (section 7, **TODO**).
- **Unseen results are noisy:** one seed, `val_unseen` has 227 images and 6 pairs, `test_unseen` has 10 pairs. Differences of a few points between backbones are not established.
- **Not run:** the optional ablations (Mixup/CutMix + EMA, class weights or a balanced sampler, an approach aimed at the crop shortcut).
