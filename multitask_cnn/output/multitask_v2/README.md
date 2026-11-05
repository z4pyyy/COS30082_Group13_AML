# Multi-task CNN on DPD_Small_clean_v2: frozen and fine-tuned ConvNeXt-Tiny

**Author:** Ariq  ·  **Date:** 2026-11-05  ·  **Track:** multitask CNN
**Run type:** main protocol runs (frozen and fine-tuned), with one documented deviation: fine-tuned batch size 32 instead of 64.

## 1. Summary

Trained one ConvNeXt-Tiny backbone with a crop head (55) and a disease head (175), scored on crop+disease pairs (333), on DPD_Small_clean_v2. Two runs: frozen backbone (batch 64, best epoch 21) and fine-tuned backbone (batch 32, best epoch 8). The best model by harmonic mean of Top-1 is the frozen run: seen 55.55 / unseen 5.46 / H 9.94 (%). The fine-tuned run gives seen 64.80 / unseen 5.25 / H 9.71 (%). Fine-tuning raises seen accuracy and Top-5 but not unseen accuracy. Unseen failure is in the disease head (7.4% frozen, 6.6% fine-tuned), not the crop head (62.9%, 66.9%).

## 2. Environment

| Item | Value |
|---|---|
| Platform | Local Windows laptop (device "Raven"), not Kaggle or Colab |
| GPU | NVIDIA GeForce GTX 1650 Ti, 4096 MiB VRAM, driver 610.88 |
| Python / torch / torchvision | Python from the project `.venv` (version not recorded) / torch 2.11.0+cu128 / torchvision not recorded |
| Dataset | DPD_Small_clean_v2, local copy at `ml project\dataset\DPD_Small_clean_v2` (same files as the Kaggle v2 dataset; Kaggle version not applicable) |
| Repo commit | 8aaaf72 (base) plus the v2 changes described in section 4: `src/common.py`, `src/train_multitask.py`, `src/plot_curves.py` |
| Notebook | `multitask_v2_kaggle.ipynb` was prepared but not used: all runs were local |

## 3. Exactly how to reproduce

Run from `multitask_cnn/` with the v2 patch applied. The dataset folder must be reachable at `DATA_ROOT` in `src/common.py`.

```powershell
# Frozen backbone, batch 64
python -m src.train_multitask --epochs 30 --batch-size 64 --run convnext_mt_frozen_v2
python -m src.evaluate_v2 --run convnext_mt_frozen_v2 --name ConvNeXt-Tiny-frozen --out output\multitask_v2
python -m src.plot_curves --run convnext_mt_frozen_v2 --name ConvNeXt-Tiny-frozen --out output\multitask_v2

# Fine-tuned backbone, batch 32 (deviation, see section 4)
python -m src.train_multitask --epochs 30 --batch-size 32 --finetune --run convnext_mt_finetune_v2
python -m src.evaluate_v2 --run convnext_mt_finetune_v2 --name ConvNeXt-Tiny-finetune --out output\multitask_v2
python -m src.plot_curves --run convnext_mt_finetune_v2 --name ConvNeXt-Tiny-finetune --out output\multitask_v2
```

Default values used: seed 42, AdamW lr 1e-3 for the heads, lr 1e-4 for the backbone when fine-tuning, weight decay 0.05, cosine schedule, mixed precision on CUDA, `--workers 4`, `--patience 10` (on val_H). The per-epoch history and config were copied into the output folder as `<name>_history.csv` and `<name>_config.json`.

## 4. Pipeline

- **Data:** `get_loaders()` from the dataset. Splits: train 6,995 / val 773 / val_unseen 227 / test_seen 2,000 / test_unseen 1,448. Nothing different in the data.
- **Augmentation:** the dataset loader's train augmentation (RandomResizedCrop, flips, rotation, colour jitter, random erasing). Nothing added.
- **Sampler / class weights:** none.
- **Model:** torchvision `convnext_tiny` with ImageNet-1k weights (`weights="DEFAULT"`). LayerNorm and Flatten kept, the ImageNet classifier removed. Two linear heads on the 768-d feature: crop (55) and disease (175). Pair score = log P(crop) + log P(disease) over all 333 pairs.
- **Loss:** unweighted cross-entropy on crop plus cross-entropy on disease. No label smoothing (matches the multitask handoff, not the shared-protocol default of 0.1).
- **Optimiser and schedule:** AdamW, weight decay 0.05, cosine schedule over the full run (no warm-up). Heads lr 1e-3. Frozen run: backbone frozen and in eval mode. Fine-tuned run: backbone lr 1e-4.
- **Epochs / patience / batch / seed:** up to 30 / 10 / 64 (frozen) and 32 (fine-tuned) / 42.
- **Model selection:** best epoch = highest val_H, the harmonic mean of val and val_unseen pair Top-1. Test sets are used once, in `evaluate_v2`.
- **Changes from the shared protocol:**
  1. **Fine-tuned run batch size 32 instead of 64.** The GTX 1650 Ti has 4 GB of VRAM. The fine-tuned backbone at batch 64 was not tested. Batch 32 was run, and it fit (peak VRAM not recorded). The frozen run uses batch 64 as specified.
  2. **Data path:** `DATA_ROOT` in `src/common.py` points at the local folder `ml project\dataset\DPD_Small_clean_v2` rather than `multitask_cnn\dataset\`. The data is the same.
  3. **No label smoothing** (handoff setting).

## 5. Process log

| # | What was run | Result | Decision |
|---|---|---|---|
| 1 | Pipeline check on the laptop, fine-tuned, 1 epoch, 5 batches, batch 32, CUDA (`runs\smoke_ft_v2`) | Ran; wrote all files | Deleted the smoke folder. Not reported as a result |
| 2 | Frozen run, 30 epochs, batch 64, CUDA | Best epoch 21, val_H 2.60%. All 30 epochs ran (no early stop) | Kept. Results in section 7 |
| 3 | Fine-tuned run started from the `ml project` root, which still held the **v1** code (`DPD_Small_clean`, best epoch by val top-1 only, no val_unseen) | 30 epochs on CUDA, batch 32. Log ends `best val comp top-1 = 0.8541`, which is the v1 format. **Not a v2 result** | Discarded. Kept as `runs\convnext_mt_finetune_v1code_DISCARD` and `logs\ft_v2.log` for the record only |
| 4 | Fine-tuned run, v2 code and v2 data, 30 epochs max, batch 32, CUDA | Early stop at epoch 18 (no val_H gain for 10 epochs). Best epoch 8, val_H 14.47% | Kept. Results in section 7 |

## 6. Training behaviour

| Model | Best epoch | Epochs run | Time / epoch (mean) | Total time | Final train loss | Val Top-1 (best epoch) | Val-unseen Top-1 (best epoch) |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny-frozen | 21 | 30 | 216 s | 1.8 h | 0.655 | 74.00 | 1.32 |
| ConvNeXt-Tiny-finetune | 8 | 18 | 540 s | 2.7 h | 0.024 | 82.92 | 7.93 |

- **Frozen:** val Top-1 rose to 74% and stayed close to that. Val-unseen stayed under 2% through most of the run. val_H never rose above 2.6%, so the best epoch is a weak choice.
- **Fine-tuned:** val Top-1 reached about 85% by epoch 18. Val-unseen rose to about 8% at epoch 8 and then fell back to about 4% by epoch 18. Training loss fell to 0.02, so the backbone is overfitting to the seen pairs. Epoch 8 was chosen by val_H, which is why its seen accuracy is lower than the final epochs.
- Curves: `ConvNeXt-Tiny-frozen_curves.png`, `ConvNeXt-Tiny-finetune_curves.png`.

## 7. Results (test sets, used once)

All values in %, Top-1 / Top-5 from `summary.csv` and `results.json`.

| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | Seen excl. held-out Top-1 | Seen per-class Top-1 | Unseen per-class Top-1 |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny-frozen | 55.55 | 5.46 | 9.94 | 81.05 | 31.49 | 45.36 | 57.27 | 35.86 | 6.21 |
| ConvNeXt-Tiny-finetune | 64.80 | 5.25 | 9.71 | 87.20 | 35.91 | 50.87 | 66.75 | 47.03 | 5.40 |

Heads (Top-1, %):

| Model | Crop head seen | Crop head unseen | Disease head seen | Disease head unseen |
|---|---|---|---|---|
| ConvNeXt-Tiny-frozen | 85.20 | 62.85 | 57.30 | 7.39 |
| ConvNeXt-Tiny-finetune | 89.30 | 66.92 | 66.05 | 6.56 |

**Compared with v1** (v1 30-epoch frozen run, seen top-1 58.0 / unseen 3.0 / H 5.8 from the v1 report; v2 frozen seen excluding held-out pairs 57.27):
- v1 frozen H Top-1 5.8 → v2 frozen 9.9. Unseen Top-1 3.0 → 5.5.
- v1 seen 58.0 → v2 seen excluding held-out pairs 57.3. The held-out pairs are removed from training in v2, so the comparable seen figure is the excluding one.

## 8. Error analysis

- **The disease head is the bottleneck on unseen pairs.** Crop head unseen accuracy is 63 to 67%, but disease head unseen accuracy is 7 to 8% for the frozen and fine-tuned runs. A correct pair needs both heads right, so the pair score stays low.
- Pair Top-1 on unseen pairs is 5.5% and Top-5 is 31 to 36%, so the correct pair is often in the top five but not the top one.
- Fine-tuning helps the crop head (89% seen, 67% unseen) and the disease head on seen pairs (66% vs 57%), but not the disease head on unseen pairs.
- Per-pair detail is in `ConvNeXt-Tiny-*_unseen_confusion_matrix.csv` and `ConvNeXt-Tiny-*_predictions_unseen.csv`. This draft does not yet list which wrong pairs dominate; that needs a pass over those files before the report.

## 9. Checkpoints

Not in git (`.gitignore` excludes `*.pt`; GitHub rejects files over 100 MB). To be sent to the team WhatsApp group as documents, right after the "done" message.

| File | Size | SHA-256 | Epoch | Val H (%) | Sent to WhatsApp group |
|---|---|---|---|---|---|
| ConvNeXt-Tiny-frozen_best.pt | 112,047,125 bytes (about 112 MB) | A1B3B1C04301C97E48D83633B5F27DB2F1392D67F07752F8A94134D7CCD31E9C | 21 | 2.60 | pending |
| ConvNeXt-Tiny-finetune_best.pt | 112,047,125 bytes (about 112 MB) | 4EB0C82F9150785D7D88FEDE218AB02B7694CC4084F5889B58BA5D9647D44F24 | 8 | 14.47 | pending |

How to load:

```python
import torch
from src.models import MultiTaskCNN
st = torch.load("ConvNeXt-Tiny-finetune_best.pt", map_location="cpu", weights_only=False)
model = MultiTaskCNN(55, 175, st["args"]["backbone"], pretrained=False)
model.load_state_dict(st["model"])
```

## 10. Known issues and open questions

- **Fine-tuned best epoch is early and unstable.** Epoch 8 was picked on val_H, which is driven by a val_unseen set of 227 images from 6 pairs. Val_unseen moves by several points between epochs.
- **Frozen run never reached a usable val_H** (2.6% at best), so its selection is weak. The results are still valid for the protocol.
- **Fine-tuned batch 32** is a documented deviation from the shared batch 64.
- **The v1-code fine-tuned run** (`convnext_mt_finetune_v1code_DISCARD`) must not be used or cited as a v2 result.
- **Python and torchvision versions** were not recorded on the laptop.
- The "next: python -m src.evaluate" line printed at the end of training is an old message from the v1 script. It doesn't run anything.
