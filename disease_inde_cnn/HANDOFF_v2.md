# Handoff: independent disease CNN on DPD_Small_clean_v2

> **For AI agents** working for this track's owner. Follow it exactly; the short human version is [`README.md`](README.md).

> **Workflow:** follow [`V2_WORKFLOW.md`](../V2_WORKFLOW.md) phase by phase (it has the expected output of every step and the sync points). **Do not run any data preparation**: the dataset is prebuilt on Kaggle; `V2_CHANGES.md` section 7 (inside the dataset) describes what each preparation script did.

**You own:** the independent **disease** classifier (175 classes), trained with 3 backbones (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny).
**Dataset:** `DPD_Small_clean_v2` (Kaggle, v2 version). v1 stays untouched; do not mix the two.
**Before you start, read:** the dataset's `README.md` (sections "Quick start" and "Shared protocol") and `V2_CHANGES.md`.
**GPU time:** about 1.5 h on a T4 for all 3 backbones.

The other two tracks (independent crop CNN, multitask CNN) follow the same protocol and output format, so the three results go straight into one table in the report.

---

## 1. What changes compared with your v1 run

| | v1 disease run | v2 |
|---|---|---|
| Validation | own 10% stratified split of train (761) | official `val.csv` (773) **and** `val_unseen.csv` (227) |
| Best epoch | best val Top-1 | best harmonic mean `H_val` of `val` and `val_unseen` Top-1 |
| Augmentation | own (RandomResizedCrop 0.5–1, rotation 25°, stronger jitter) | the dataset loader's augmentation |
| Mixup / CutMix, EMA | on | **off** in the main run (allowed as a separate run, section 5) |
| Data loading | own `DiseaseDS` class | `get_loaders()` from the dataset |
| Pretrained weights | timm defaults (`convnext_tiny` = ImageNet-12k pretraining) | pinned ImageNet-1k tags, same as the other tracks |
| Train | 6,439 | 6,995 |
| Reported metrics | Top-1/3/5, H, confusion matrices, clean train accuracy | the same, plus val_unseen, per-class accuracy, seen excluding held-out pairs, per-image predictions |

Kept: AdamW (wd 0.05), OneCycle, label smoothing 0.1, batch 64, seed 42, 30 epochs, patience 10, learning rates 3e-4 / 5e-4 / 1e-4, and clean train accuracy.

Why the extras are off: the crop and multitask tracks do not use them. With them on, a difference between tracks could come from Mixup/EMA rather than from the task. Your v1 results showed they did not fix unseen accuracy anyway (9.8% for ConvNeXt).

---

## 2. Run it

The dataset ships a reference script, `train_independent.py`, that already implements the protocol and writes every required output file. **Use it as is for the main run.**

### Kaggle (recommended)

1. New notebook → Add data → the team's `DPD_Small_clean_v2` dataset (v2 version).
2. Settings → Accelerator → **GPU T4** (or P100). Internet **on** (pretrained weights download).
3. Cells:

```python
!pip -q install timm
import glob, os
DATA = os.path.dirname(glob.glob("/kaggle/input/**/label_maps.json", recursive=True)[0])
print(DATA)
```

```python
!python {DATA}/train_independent.py --task disease --out /kaggle/working/disease_independent_v2 --workers 4
```

4. Download `/kaggle/working/disease_independent_v2/` when it finishes (`!cd /kaggle/working && zip -r disease_independent_v2.zip disease_independent_v2`).

### Colab

```python
from google.colab import drive; drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/<path>/DPD_Small_clean_v2" /content/      # local copy: Drive is slow to read
!pip -q install timm
!python /content/DPD_Small_clean_v2/train_independent.py --task disease --workers 2 \
        --out /content/drive/MyDrive/DPD_runs/v2/disease_independent_v2
```

`results.json` is rewritten after each backbone, so finished backbones survive a disconnect.

### Quick check before the long run (optional, 2 minutes)

```python
!python {DATA}/train_independent.py --task disease --out /kaggle/working/smoke --models ResNet50 --epochs 1 --limit-batches 3
```

---

## 3. What to deliver

### 3.1 In the repo: `disease_inde_cnn/output/disease_independent_v2/`

Note the new `output/` level: v1 results sit at `disease_inde_cnn/disease_independent/`; v2 goes under `output/` to match the crop and multitask folders.

```
disease_inde_cnn/
├── disease_independent_cnns_v2.ipynb       the notebook you ran, WITH outputs (logs visible)
└── output/
    └── disease_independent_v2/
        ├── README.md                       filled from README_TEMPLATE.md (how you trained, the whole process)
        ├── config.json                     written by the script
        ├── results.json                    written by the script (schema below)
        ├── summary.csv                     one row per backbone
        ├── all_models_accuracy.png
        ├── harmonic_mean.png
        ├── <Model>_history.csv             per epoch: losses, train acc, val, val_unseen, H_val, seconds
        ├── <Model>_curves.png
        ├── <Model>_<split>_confusion_matrix.csv / .png     split = val, val_unseen, seen, unseen
        └── <Model>_predictions_<split>.csv                 per image: path, true, pred1..pred5
```

`<Model>` is `ResNet50`, `EfficientNet-B0`, `ConvNeXt-Tiny`. **Do not edit or delete the v1 folder** `disease_inde_cnn/disease_independent/`.

### 3.2 The `best.pt` checkpoints

The script saves `<Model>_best.pt` (about 30–110 MB each). **They cannot go into git**: `.gitignore` excludes `*.pt`, and GitHub rejects files over 100 MB. Instead, share them through the team WhatsApp group:

1. Push your results folder and notebook to the repo first (section 3.1), so the message can point at the commit.
2. **Post in the team WhatsApp group** that the work is done and what was done. Use this format:

   ```
   [v2 disease_independent] DONE
   What: independent disease CNN, 3 backbones, shared v2 protocol
   Best (Top-1): <best backbone> – seen xx.x / unseen xx.x / H xx.x
   Pushed: disease_inde_cnn/output/disease_independent_v2/ (commit <hash>)
   Checkpoints: ResNet50_best.pt, EfficientNet-B0_best.pt, ConvNeXt-Tiny_best.pt (sent below; SHA-256 in README)
   Notes: <anything off-protocol, problems, or "none">
   ```

3. **Right after that message, send the `.pt` files to the same group.** Attach them as **Document** (not as media), so WhatsApp sends the file unchanged. WhatsApp allows up to 2 GB per file, so every checkpoint fits.
4. In your `README.md`, list each file with its size, SHA-256 (`!sha256sum /kaggle/working/disease_independent_v2/*.pt`) and the date it was sent to the group. Anyone who downloads a file from the group can check it against that SHA-256.

Reload with:

```python
ck = torch.load("ConvNeXt-Tiny_best.pt", map_location="cpu")
model = timm.create_model(ck["timm_id"], pretrained=False, num_classes=ck["num_classes"])
model.load_state_dict(ck["state_dict"])
```

### 3.3 `README.md` for your run

Copy [`README_TEMPLATE.md`](README_TEMPLATE.md) to `output/disease_independent_v2/README.md` and fill in every section: environment, exact commands, timings, everything you tried (including failed runs and why), the results tables, what you see in the curves and confusion matrices, the checkpoint list (size, SHA-256, date sent), and anything you changed from the protocol. Someone else should be able to reproduce your numbers from it alone.

### 3.4 `results.json` schema (written by the script)

```json
{
  "ConvNeXt-Tiny": {
    "timm_id": "convnext_tiny.fb_in1k", "lr": 0.0001,
    "best_epoch": 0, "epochs_run": 0, "selection": "max H(val, val_unseen) Top-1",
    "train_clean_top1": 0,
    "val":        {"1": 0, "3": 0, "5": 0, "per_class_top1": 0},
    "val_unseen": {"1": 0, "3": 0, "5": 0, "per_class_top1": 0},
    "seen":       {"1": 0, "3": 0, "5": 0, "per_class_top1": 0},
    "seen_excl_heldout": {"1": 0, "3": 0, "5": 0},
    "unseen":     {"1": 0, "3": 0, "5": 0, "per_class_top1": 0},
    "harmonic":   {"1": 0, "3": 0, "5": 0, "per_class_top1": 0}
  }
}
```

All values in %. `seen` = `test_seen`, `unseen` = `test_unseen`.

---

## 4. Checks before you push

- [ ] The log's first line shows split sizes train 6995, val 773, val_unseen 227, test_seen 2000, test_unseen 1448.
- [ ] `config.json` shows `"dataset": "DPD_Small_clean_v2"`, the pinned `timm_id`s and a GPU name.
- [ ] Every backbone has all 4 confusion matrices and all 4 prediction files.
- [ ] `summary.csv` has 3 rows.
- [ ] `README.md` is filled in, with checkpoint sizes and SHA-256.
- [ ] After pushing: "done" message posted in the WhatsApp group, then the `.pt` files sent there as documents.
- [ ] The notebook is saved with outputs as `disease_independent_cnns_v2.ipynb`.

---

## 5. Your own improvements (optional, separate runs)

The main run must use the script unchanged. Then you can try ideas, each in its own folder with the same files, e.g.:
- `output/disease_independent_v2_mixup_ema/`: your v1 Mixup/CutMix + EMA, to show whether they help under the v2 protocol;
- class weights (`class_weights(col="disease_id")` from `dataset.py`) or a balanced sampler (`get_loaders(balance_by="disease_id")`), not both;
- anything aimed at the crop shortcut (section 6).

Tune on `val` / `val_unseen` only. Copy `train_independent.py` into `disease_inde_cnn/` and edit your copy; do not edit the dataset's copy. Describe each run in its own `README.md`.

---

## 6. What to look at for the report

This track is where the dataset's main problem shows. From the v1 analysis:
- **Disease is tied to crop.** 77% of diseases appear on only one crop in `train`. In v1, 52–60% of wrong unseen predictions were a disease the same crop had in training: tomato early_blight → septoria / late_blight; grape black_rot → esca; cotton bacterial_blight → target_spot.
- **Only powdery_mildew transferred** (59–86% across all v1 pipelines). It is the one unseen disease spread over many crops in training.
- **Check `val_unseen` per epoch** in `<Model>_history.csv`: if it stays near 0 while `val` climbs, the model is learning crop → disease, not symptoms.
- **Use the metadata:** `disease_n_train_crops` in the CSVs. Accuracy broken down by how many crops a disease appears on in training is strong evidence for the report. Compute it from `<Model>_predictions_unseen.csv` joined with `test_unseen.csv` on `path`.
- **v1 vs v2:** compare `seen_excl_heldout` with v1 seen, not `seen`.
