# Handoff: multitask CNN on DPD_Small_clean_v2

> **For AI agents** working for this track's owner. Follow it exactly; the short human version is [`README.md`](README.md).

> **Workflow:** follow [`V2_WORKFLOW.md`](../V2_WORKFLOW.md) phase by phase (it has the expected output of every step and the sync points). **Do not run any data preparation**: the dataset is prebuilt on Kaggle; `V2_CHANGES.md` section 7 (inside the dataset) describes what each preparation script did.

**You own:** the multitask CNN: one ConvNeXt-Tiny backbone with a crop head (55) and a disease head (175), scored on crop+disease pairs (333).
**Dataset:** `DPD_Small_clean_v2` (Kaggle, v2 version). v1 stays untouched; do not mix the two.
**Before you start, read:** the dataset's `README.md` (sections "Quick start" and "Shared protocol") and `V2_CHANGES.md`.
**GPU time:** roughly 1.5 h per run, 3 h for both. Your v1 frozen run took about 190 s per epoch; with early stopping, a run whose best epoch is around 17 stops near epoch 27. Fine-tuning has more to compute per batch but is often limited by image loading, so time the first epoch and plan from that.

Your v1 pipeline was already the one that followed the team rules (shared loader, official `val.csv`, tests used once). v2 keeps your code and adds `val_unseen`, best-epoch selection on the harmonic mean, early stopping, and the output format shared with the two independent tracks.

---

## 1. What changes compared with your v1 run

| | v1 multitask run | v2 |
|---|---|---|
| Data | `DPD_Small_clean` | `DPD_Small_clean_v2` (same images; train 6,995, val 773, new `val_unseen` 227) |
| Best epoch | best val pair Top-1 | best harmonic mean `val_H` of `val` and `val_unseen` pair Top-1 |
| Early stopping | none (all epochs run) | patience 10 on `val_H` |
| Runs | frozen backbone only | **frozen** (comparable with v1) **and fine-tuned** (comparable with the independent tracks, which fine-tune) |
| Outputs | `metrics.json`, predictions, seen confusion matrix | shared v2 format (below) via `src/evaluate_v2.py`, plus your existing `evaluate.py` outputs if you like |

Unchanged: torchvision ConvNeXt-Tiny ImageNet-1k weights (the same as the independent tracks' pinned `convnext_tiny.fb_in1k`), unweighted CE(crop) + CE(disease), AdamW lr 1e-3 (heads) and lr/10 for the backbone when fine-tuning, wd 0.05, cosine schedule, batch 64, seed 42, 30 epochs.

---

## 2. Code changes (already prepared, tested)

Two things are in your folder:

1. **`v2_protocol.patch`**: the training changes. Points `src/common.py` at `DPD_Small_clean_v2`, and in `src/train_multitask.py` evaluates `val_unseen` every epoch, logs `unseen_*` and `val_H` columns, keeps `best.pt` by `val_H`, and adds `--patience` (default 10). Apply it from the repo root:

   ```
   git apply multitask_cnn/v2_protocol.patch
   ```

   After this, the code needs v2. To re-run a v1 experiment, revert with `git apply -R multitask_cnn/v2_protocol.patch`.

2. **`src/evaluate_v2.py`** (new file): writes the shared v2 output files and `results.json` / `summary.csv` from a run's `best.pt`, for the pair scores and for each head separately. Your existing `src/evaluate.py` still works and can be kept for your own `metrics.json` and `report.md`.

Both were smoke-tested on CPU against the v2 dataset (random weights, a few batches): the training log gets the new columns, early stopping triggers, and `evaluate_v2` writes every file listed in section 4.

---

## 3. Run it

### Data location

Your code expects the dataset at `multitask_cnn/dataset/DPD_Small_clean_v2/`. On Kaggle, link it there:

```python
import glob, os
DATA = os.path.dirname(glob.glob("/kaggle/input/**/label_maps.json", recursive=True)[0])
!mkdir -p /kaggle/working/repo/multitask_cnn/dataset
!ln -sfn {DATA} /kaggle/working/repo/multitask_cnn/dataset/DPD_Small_clean_v2
```

(Clone or upload the repo to `/kaggle/working/repo` first. On Colab, copy the dataset folder to that path instead; reading from Drive is slow.)

### Commands (from `multitask_cnn/`)

```
# 1. frozen backbone (comparable with your v1 baseline)
python -m src.train_multitask --epochs 30 --run convnext_mt_frozen_v2
python -m src.evaluate_v2 --run convnext_mt_frozen_v2 --name ConvNeXt-Tiny-frozen --out output/multitask_v2

# 2. fine-tuned backbone (comparable with the independent tracks)
python -m src.train_multitask --epochs 30 --finetune --run convnext_mt_finetune_v2
python -m src.evaluate_v2 --run convnext_mt_finetune_v2 --name ConvNeXt-Tiny-finetune --out output/multitask_v2
```

`evaluate_v2` adds a row to `results.json` and `summary.csv` each time, so both runs end up in one table. Use `--workers 4` on Kaggle and `2` on Colab.

Then copy each run's log and curves next to the results:

```
cp runs/convnext_mt_frozen_v2/train_log.csv   output/multitask_v2/ConvNeXt-Tiny-frozen_history.csv
cp runs/convnext_mt_finetune_v2/train_log.csv output/multitask_v2/ConvNeXt-Tiny-finetune_history.csv
cp runs/convnext_mt_frozen_v2/config.json     output/multitask_v2/ConvNeXt-Tiny-frozen_config.json
cp runs/convnext_mt_finetune_v2/config.json   output/multitask_v2/ConvNeXt-Tiny-finetune_config.json
```

and save a curves plot per run as `<name>_curves.png` (pair Top-1 on `val` and `val_unseen`, `val_H`, train loss; mark the best epoch).

Optional, if time allows: ResNet-50 and EfficientNet-B0 backbones would need adding to `build_backbone()` in `src/models.py`. Use torchvision `resnet50(weights="DEFAULT")` and `efficientnet_b0(weights="DEFAULT")`. Name the rows `ResNet50-finetune` etc.

---

## 4. What to deliver

### 4.1 In the repo: `multitask_cnn/output/multitask_v2/`

```
multitask_cnn/
├── v2_protocol.patch                        applied (commit the changed src/ files)
├── src/evaluate_v2.py
└── output/
    └── multitask_v2/
        ├── README.md                        filled from README_TEMPLATE.md (how you trained, the whole process)
        ├── results.json                     written by evaluate_v2 (schema below)
        ├── summary.csv                      one row per run
        ├── <name>_config.json               copied from runs/<run>/config.json
        ├── <name>_history.csv               copied from runs/<run>/train_log.csv
        ├── <name>_curves.png
        ├── <name>_<split>_confusion_matrix.csv / .png           pair (333 classes)
        ├── <name>_predictions_<split>.csv                       path, true, pred1..pred5, crop_head_pred, disease_head_pred
        ├── <name>_crop_<split>_confusion_matrix.csv / .png      crop head
        ├── <name>_crop_predictions_<split>.csv
        ├── <name>_disease_<split>_confusion_matrix.csv / .png   disease head
        └── <name>_disease_predictions_<split>.csv
```

`<name>` is `ConvNeXt-Tiny-frozen`, `ConvNeXt-Tiny-finetune`; `<split>` is `val`, `val_unseen`, `seen`, `unseen`. This matches the independent tracks' layout (`crop_indi_cnn/output/crop_independent_v2/`, `disease_inde_cnn/output/disease_independent_v2/`), so the report can read all three the same way.

**Do not edit or delete** the v1 folder `output/convnext_mt_baseline_30ep/`.

### 4.2 The `best.pt` checkpoints

`runs/<run>/best.pt` (about 110 MB for ConvNeXt-Tiny). **They cannot go into git**: `.gitignore` excludes `*.pt` and `runs/`, and GitHub rejects files over 100 MB. Instead, share them through the team WhatsApp group:

Rename each `runs/<run>/best.pt` to `<name>_best.pt` (`ConvNeXt-Tiny-frozen_best.pt`, `ConvNeXt-Tiny-finetune_best.pt`), then:

1. Push your results folder and the changed `src/` files to the repo first (section 4.1), so the message can point at the commit.
2. **Post in the team WhatsApp group** that the work is done and what was done. Use this format:

   ```
   [v2 multitask] DONE
   What: multitask CNN, ConvNeXt-Tiny frozen + fine-tuned, shared v2 protocol
   Best (Top-1): pair, <frozen | finetune> – seen xx.x / unseen xx.x / H xx.x
   Pushed: multitask_cnn/output/multitask_v2/ (commit <hash>)
   Checkpoints: ConvNeXt-Tiny-frozen_best.pt, ConvNeXt-Tiny-finetune_best.pt (sent below; SHA-256 in README)
   Notes: <anything off-protocol, problems, or "none">
   ```

3. **Right after that message, send the `.pt` files to the same group.** Attach them as **Document** (not as media), so WhatsApp sends the file unchanged. WhatsApp allows up to 2 GB per file, so every checkpoint fits.
4. In your `README.md`, list each file with its size, SHA-256 (`sha256sum *_best.pt`) and the date it was sent to the group. Anyone who downloads a file from the group can check it against that SHA-256.

Each checkpoint holds `model` (state dict), `args`, `epoch`, `val`, `val_unseen`, `val_H`. Reload with:

```python
st = torch.load("ConvNeXt-Tiny-finetune_best.pt", map_location="cpu", weights_only=False)
model = MultiTaskCNN(55, 175, st["args"]["backbone"], pretrained=False); model.load_state_dict(st["model"])
```

### 4.3 `README.md` for your runs

Copy [`README_TEMPLATE.md`](README_TEMPLATE.md) to `output/multitask_v2/README.md` and fill in every section: environment, exact commands, timings, everything you tried (including failed runs and why), the results tables, what you see in the curves and confusion matrices, the checkpoint list (size, SHA-256, date sent), and anything you changed from the protocol.

### 4.4 `results.json` schema (written by `evaluate_v2`)

```json
{
  "ConvNeXt-Tiny-finetune": {
    "backbone": "convnext_tiny", "finetune": true, "run": "convnext_mt_finetune_v2",
    "best_epoch": 0, "selection": "max H(val, val_unseen) pair Top-1",
    "pair": {
      "val": {"1": 0, "3": 0, "5": 0, "per_class_top1": 0},
      "val_unseen": {"...": 0}, "seen": {"...": 0}, "seen_excl_heldout": {"1": 0, "3": 0, "5": 0},
      "unseen": {"...": 0}, "harmonic": {"1": 0, "3": 0, "5": 0, "per_class_top1": 0}
    },
    "crop_head":    {"same keys as pair": 0},
    "disease_head": {"same keys as pair": 0}
  }
}
```

All values in %. `seen` = `test_seen`, `unseen` = `test_unseen`. The pair is the headline metric; the heads explain it.

---

## 5. Checks before you push

- [ ] `git apply --check multitask_cnn/v2_protocol.patch` was clean, and the changed `src/` files are committed.
- [ ] The training log has `unseen_comp_top1` and `val_H` columns.
- [ ] `results.json` and `summary.csv` have both runs.
- [ ] Every run has the pair, crop-head and disease-head confusion matrices and predictions for all 4 splits.
- [ ] `README.md` is filled in, with checkpoint sizes and SHA-256.
- [ ] After pushing: "done" message posted in the WhatsApp group, then the `.pt` files sent there as documents.

---

## 6. Your own improvements (optional, separate runs)

Keep the two protocol runs above unchanged. Ideas go in separately named runs and folders (e.g. `output/multitask_v2_weighted/`), tuned on `val` / `val_unseen` only, each with its own `README.md`. Natural ones for this track: loss weighting between the heads, class weights (`class_weights(col="disease_id")` from the dataset) or a balanced sampler (`get_loaders(balance_by="comp_id")`), not both.

---

## 7. What to look at for the report

From the v1 analysis (expect similar patterns, check whether they hold):
- **The disease head is the bottleneck on unseen pairs.** In v1: crop head 62.2% vs disease head 5.1% on `test_unseen`, pair 3.0%. The wrong pair is usually a disease the model already knows on the same crop (tomato early blight → tomato late blight; grape black rot → grape esca).
- **Frozen vs fine-tuned** shows how much of the gap to the independent tracks is down to the frozen backbone.
- **Pair vs independent tracks:** the independent crop and disease tracks save per-image predictions (`<Model>_predictions_<split>.csv`), so the report can compare your pair accuracy with simply combining their two predictions.
- **v1 vs v2:** compare `seen_excl_heldout` with v1 seen, not `seen`.
