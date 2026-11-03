# Handoff: independent crop CNN on DPD_Small_clean_v2

> **For AI agents** working for this track's owner. Follow it exactly; the short human version is [`README.md`](README.md).

> **Workflow:** follow [`V2_WORKFLOW.md`](../V2_WORKFLOW.md) phase by phase (it has the expected output of every step and the sync points). **Do not run any data preparation**: the dataset is prebuilt on Kaggle; `V2_CHANGES.md` section 7 (inside the dataset) describes what each preparation script did.

**You own:** the independent **crop** classifier (55 classes), trained with 3 backbones (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny).
**Dataset:** `DPD_Small_clean_v2` (Kaggle, v2 version). v1 stays untouched; do not mix the two.
**Before you start, read:** the dataset's `README.md` (sections "Quick start" and "Shared protocol") and `V2_CHANGES.md`.
**GPU time:** about 1.5 h on a T4 for all 3 backbones.

The other two tracks (independent disease CNN, multitask CNN) follow the same protocol and output format, so the three results go straight into one table in the report.

---

## 1. What changes compared with your v1 run

| | v1 crop run | v2 |
|---|---|---|
| Early stopping / best epoch | on **`test_seen`** (a test leak) | on `val` + `val_unseen`: best epoch = highest harmonic mean `H_val` |
| Augmentation | **none** (`tf_train = tf_eval`); hit 100% train accuracy by epoch 7 | the dataset loader's augmentation (train only) |
| Data loading | own `CropDS` class | `get_loaders()` from the dataset |
| Pretrained weights | timm defaults (`convnext_tiny` = ImageNet-12k pretraining) | pinned ImageNet-1k tags, same as the other tracks |
| Train / val | 7,200 / test_seen | 6,995 / 773 + 227 (`val_unseen`) |
| Reported metrics | Top-1/3/5, H, confusion matrices | the same, plus per-class accuracy, seen excluding held-out pairs, clean train accuracy, per-image predictions |

Everything else stays: AdamW (wd 0.05), OneCycle, label smoothing 0.1, batch 64, seed 42, 30 epochs, patience 10, and the learning rates 3e-4 / 5e-4 / 1e-4.

---

## 2. Run it

The dataset ships a reference script, `train_independent.py`, that already implements the protocol and writes every required output file. **Use it as is for the main run.** If you want to try your own improvements, see section 5.

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
!python {DATA}/train_independent.py --task crop --out /kaggle/working/crop_independent_v2 --workers 4
```

4. When it finishes, download `/kaggle/working/crop_independent_v2/` (Output tab, or zip it: `!cd /kaggle/working && zip -r crop_independent_v2.zip crop_independent_v2`).

### Colab

```python
from google.colab import drive; drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/<path>/DPD_Small_clean_v2" /content/      # local copy: Drive is slow to read
!pip -q install timm
!python /content/DPD_Small_clean_v2/train_independent.py --task crop --workers 2 \
        --out /content/drive/MyDrive/DPD_runs/v2/crop_independent_v2
```

Writing `--out` to Drive keeps the results if the session disconnects. `results.json` is rewritten after each backbone, so finished backbones are never lost.

### Quick check before the long run (optional, 2 minutes)

```python
!python {DATA}/train_independent.py --task crop --out /kaggle/working/smoke --models ResNet50 --epochs 1 --limit-batches 3
```

---

## 3. What to deliver

### 3.1 In the repo: `crop_indi_cnn/output/crop_independent_v2/`

```
crop_indi_cnn/
├── crop_independent_cnns_v2.ipynb          the notebook you ran, WITH outputs (logs visible)
└── output/
    └── crop_independent_v2/
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

`<Model>` is `ResNet50`, `EfficientNet-B0`, `ConvNeXt-Tiny`. This is the same layout as the v1 folder `output/crop_independent/`, plus `README.md`, `config.json`, the `val` / `val_unseen` files and the prediction files.

**Do not edit or delete the v1 folder** `output/crop_independent/`. It is the v1 baseline for comparison.

### 3.2 The `best.pt` checkpoints

The script saves `<Model>_best.pt` (about 30–110 MB each) in the output folder. **They cannot go into git**: the repo's `.gitignore` excludes `*.pt`, and GitHub rejects files over 100 MB. Instead, share them through the team WhatsApp group:

1. Push your results folder and notebook to the repo first (section 3.1), so the message can point at the commit.
2. **Post in the team WhatsApp group** that the work is done and what was done. Use this format:

   ```
   [v2 crop_independent] DONE
   What: independent crop CNN, 3 backbones, shared v2 protocol
   Best (Top-1): <best backbone> – seen xx.x / unseen xx.x / H xx.x
   Pushed: crop_indi_cnn/output/crop_independent_v2/ (commit <hash>)
   Checkpoints: ResNet50_best.pt, EfficientNet-B0_best.pt, ConvNeXt-Tiny_best.pt (sent below; SHA-256 in README)
   Notes: <anything off-protocol, problems, or "none">
   ```

3. **Right after that message, send the `.pt` files to the same group.** Attach them as **Document** (not as media), so WhatsApp sends the file unchanged. WhatsApp allows up to 2 GB per file, so every checkpoint fits.
4. In your `README.md`, list each file with its size, SHA-256 (`!sha256sum /kaggle/working/crop_independent_v2/*.pt`) and the date it was sent to the group. Anyone who downloads a file from the group can check it against that SHA-256.

Each checkpoint holds `state_dict`, `timm_id`, `num_classes`, `class_names`, `task`, `epoch`, `val_H`, `val_top1`, `val_unseen_top1`, `dataset`. Reload with:

```python
ck = torch.load("ConvNeXt-Tiny_best.pt", map_location="cpu")
model = timm.create_model(ck["timm_id"], pretrained=False, num_classes=ck["num_classes"])
model.load_state_dict(ck["state_dict"])
```

### 3.3 `README.md` for your run

Copy [`README_TEMPLATE.md`](README_TEMPLATE.md) to `output/crop_independent_v2/README.md` and fill in every section: environment, exact commands, timings, everything you tried (including failed or abandoned runs and why), the results tables, what you see in the curves and confusion matrices, the checkpoint list (size, SHA-256, date sent), and anything you changed from the protocol. Someone else should be able to reproduce your numbers from it alone.

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

- [ ] The log's first lines show split sizes train 6995, val 773, val_unseen 227, test_seen 2000, test_unseen 1448 (or check `len(loaders[...].dataset)`).
- [ ] `config.json` shows `"dataset": "DPD_Small_clean_v2"`, the pinned `timm_id`s and a GPU name.
- [ ] Every backbone has all 4 confusion matrices and all 4 prediction files.
- [ ] `summary.csv` has 3 rows.
- [ ] Clean train accuracy (`train_clean_top1`) is below 100%, and the curves show `train_acc_aug` rising more slowly than in v1. That shows augmentation is on.
- [ ] `README.md` is filled in, with checkpoint sizes and SHA-256.
- [ ] After pushing: "done" message posted in the WhatsApp group, then the `.pt` files sent there as documents.
- [ ] The notebook is saved with outputs as `crop_independent_cnns_v2.ipynb`.

---

## 5. Your own improvements (optional, separate runs)

The main run above must use the script unchanged, so the three tracks stay comparable. If you then try ideas (other augmentation, class weights, a different learning rate, crop-specific tricks):
- run them into a **separately named folder**, e.g. `output/crop_independent_v2_classweights/`, with the same files;
- keep tuning on `val` / `val_unseen` only, never on the test sets;
- describe each one in its own `README.md`, and compare against the main run.

To change the script, copy `train_independent.py` into `crop_indi_cnn/` and edit your copy; do not edit the dataset's copy.

---

## 6. What to look at for the report

From the v1 analysis (expect similar patterns, check whether they hold):
- **Unseen crop accuracy is flattered by tomato**, which is 33% of `test_unseen`. In v1, cherry (33%), cucumber (42%) and pepper (62%) were much worse: those crops have few training images, all of other diseases. Use the per-class numbers and `<Model>_unseen_confusion_matrix.png`.
- **Per-class vs overall:** v1 seen crop was 91.8% overall but 67.5% averaged per class (long tail).
- **v1 vs v2:** compare `seen_excl_heldout` (1,940 images) with v1 seen, not `seen`.
- `<Model>_predictions_unseen.csv` lets the multitask person and the report combine your crop predictions with the disease track's predictions per image.
