# v2 retrain: end-to-end workflow for all three tracks

One page everyone follows, step by step, so the three tracks (crop, disease, multitask) end up with results that fit into one table.

| Track | Owner | Folder | Your short guide | Your agent's guide |
|---|---|---|---|---|
| Independent crop CNN | <name> | `crop_indi_cnn/` | [README.md](crop_indi_cnn/README.md) | [HANDOFF_v2.md](crop_indi_cnn/HANDOFF_v2.md) |
| Independent disease CNN | <name> | `disease_inde_cnn/` | [README.md](disease_inde_cnn/README.md) | [HANDOFF_v2.md](disease_inde_cnn/HANDOFF_v2.md) |
| Multitask CNN | <name> | `multitask_cnn/` | [README.md](multitask_cnn/README.md) | [HANDOFF_v2.md](multitask_cnn/HANDOFF_v2.md) |
| Dataset (v2) | <name> | `data_prep_v2/` | `README.md` and `V2_CHANGES.md` inside the dataset | — |

**Each phase ends with a ✅ "you should see" check.** If you don't see it, stop and use the "If something goes wrong" table at the end, or ask in the WhatsApp group. Don't work around it silently.

---

## Phase 0: dataset is ready (dataset maintainer only)

Model developers skip this phase. **Do not run any data preparation yourself.** The dataset on Kaggle is already built and checked. What data preparation did, script by script, is in `V2_CHANGES.md` section 7 (inside the dataset), so you can describe it in the report.

Maintainer checklist:
- [ ] `python data_prep_v2/build_v2.py --raw <raw> --out <out>` and `python data_prep_v2/verify_v2.py --raw <raw> --data <out>`: all 8 checks print `ok`.
- [ ] Upload the contents of `DPD_Small_clean_v2/` to Kaggle as a **new version** of the team dataset. v1 stays as the earlier version.
- [ ] Commit and push the v2 docs and code to the repo.
- [ ] Post in the WhatsApp group:
  ```
  [v2 dataset] READY
  Kaggle: <dataset link>, version <n>
  Repo: commit <hash>
  Start with V2_WORKFLOW.md in the repo root.
  ```

✅ The group has the Kaggle link, the version number and the commit hash.

---

## Phase 1: kickoff (everyone, 10 minutes)

1. `git pull` the repo at the commit from the group message.
2. Read this file, then your track's `README.md`. If you use an AI agent, give it your track's `HANDOFF_v2.md`.
3. Read the dataset's `README.md` "Quick start" and "Shared protocol" sections.
4. **Sync point:** reply in the group:
   ```
   [v2 <track>] STARTING – Kaggle v<n>, commit <hash>
   ```

✅ All three owners have replied with **the same Kaggle version and commit**. If anyone's differs, fix it now, not after training.

**The rules that keep the tracks comparable** (all three tracks):

| Rule | Why |
|---|---|
| Use the dataset's loader (`get_loaders()`) and its augmentation unchanged | same data pipeline for every track |
| Never use `test_seen` / `test_unseen` to choose epochs, learning rates or models | they are only for the final numbers |
| Best epoch = highest harmonic mean of `val` and `val_unseen` Top-1, patience 10 | the brief's headline metric is the seen/unseen harmonic mean |
| Seed 42, batch 64, up to 30 epochs, AdamW wd 0.05 | identical settings |
| Pinned ImageNet-1k weights (`convnext_tiny.fb_in1k` etc.) | v1 tracks used different pretrained weights |
| Main run first, unchanged; your own ideas only afterwards, in a separately named folder | the main runs are what get compared |
| Don't edit or delete any v1 results folder | v1 is the baseline |

---

## Phase 2: set up the Kaggle notebook (10 minutes)

1. Kaggle → **Create → New Notebook**.
2. Right panel → **Add Input** → search the team dataset → add **version <n>** (check the version number!).
3. Right panel → **Settings**: Accelerator **GPU T4 x2** or **P100**; **Internet on** (needed to download the pretrained weights); Persistence: Files.
4. First cell, the sanity check:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install timm
import glob, os, sys
DATA = os.path.dirname(glob.glob("/kaggle/input/**/label_maps.json", recursive=True)[0])
sys.path.insert(0, DATA)
import torch, timm, dataset
print("DATA:", DATA)
print("torch", torch.__version__, "| timm", timm.__version__, "| cuda", torch.cuda.is_available())
print({s: len(dataset.read_split(s)) for s in dataset.SPLITS})
```

✅ You should see a GPU name (e.g. `Tesla T4`), `cuda True`, and exactly:
```
{'train': 6995, 'val': 773, 'val_unseen': 227, 'test_seen': 2000, 'test_unseen': 1448}
```

Multitask only: you also need the repo code in the notebook. Clone or upload the repo to `/kaggle/working/repo`, apply `multitask_cnn/v2_protocol.patch`, and link the dataset to `multitask_cnn/dataset/DPD_Small_clean_v2` (exact cells in `multitask_cnn/HANDOFF_v2.md` section 3).

---

## Phase 3: smoke test (about 2 minutes)

A tiny run that exercises the whole pipeline before you spend 1.5 hours of GPU.

Crop / disease (use your task):
```python
!python {DATA}/train_independent.py --task crop --out /kaggle/working/smoke --models ResNet50 --epochs 1 --limit-batches 3
```
Multitask (from `multitask_cnn/`):
```python
!python -m src.train_multitask --epochs 1 --limit-batches 3 --run smoke
!python -m src.evaluate_v2 --run smoke --name smoke --out /kaggle/working/smoke --limit-batches 3
```

✅ You should see:
- crop / disease: a line `split sizes: {'train': 6995, ...} | device: cuda`, one epoch line like `[ResNet50] ep 01 | train loss … | val … | val_unseen … | H …`, then a one-row summary table;
- multitask: an epoch line containing `unseen_comp_top1=` and `val_H=`, then a one-row summary table;
- `/kaggle/working/smoke/` containing `results.json`, `summary.csv`, confusion matrices and prediction files.

The accuracy numbers will be meaningless (3 batches). Only check that it runs. Then delete the smoke folder: `!rm -rf /kaggle/working/smoke`.

---

## Phase 4: the main run (1.5–3 h of GPU)

Run it as a **saved version** so it keeps going if your browser closes: put the command in a cell, then **Save Version → Save & Run All (Commit)**. Kaggle runs the notebook in the background (up to 12 h) and keeps `/kaggle/working/` as the version's output.

| Track | Command | GPU time (T4) |
|---|---|---|
| Crop | `!python {DATA}/train_independent.py --task crop --out /kaggle/working/crop_independent_v2 --workers 4` | ~1.5 h |
| Disease | `!python {DATA}/train_independent.py --task disease --out /kaggle/working/disease_independent_v2 --workers 4` | ~1.5 h |
| Multitask | frozen + fine-tuned train and `evaluate_v2` commands from `multitask_cnn/README.md` step 4 | ~3 h |

What is normal while it runs:
- One line per epoch with `val`, `val_unseen` and `H`. **`val_unseen` is small (227 images) and jumps a few points between epochs.** That's expected; the best epoch is chosen on `H`.
- **Disease and multitask:** `val_unseen` may stay low while `val` climbs. That is the dataset's known problem (diseases tied to crops), not a bug. Write it down for the report.
- Early stopping prints `early stop at epoch N` when `H` hasn't improved for 10 epochs. Runs often end before epoch 30.
- Crop / disease: `results.json` is rewritten after each backbone, so finished backbones are safe even if a later one crashes.

✅ The version finishes "Successfully", and the output contains the final summary table (3 rows for crop/disease, 2 for multitask).

---

## Phase 5: collect and check the outputs (15 minutes)

1. Open the finished version → **Output** → download the results folder (or zip it first: `!cd /kaggle/working && zip -r out.zip <folder>`).
2. Put it in the repo:

| Track | Repo folder |
|---|---|
| Crop | `crop_indi_cnn/output/crop_independent_v2/` |
| Disease | `disease_inde_cnn/output/disease_independent_v2/` (new `output/` level) |
| Multitask | `multitask_cnn/output/multitask_v2/` |

3. Count the files (excluding `.pt` and your `README.md`):

| Track | Expected | Made of |
|---|---|---|
| Crop / disease | **47** | per backbone (×3): 4 confusion-matrix CSVs + 4 PNGs + 4 prediction CSVs + history + curves = 14 → 42; plus `config.json`, `results.json`, `summary.csv`, `all_models_accuracy.png`, `harmonic_mean.png` |
| Multitask | **80** | per run (×2): pair, crop head and disease head, each 4 CSVs + 4 PNGs + 4 predictions = 36, plus history, config, curves = 39 → 78; plus `results.json`, `summary.csv` |

```bash
ls <folder> | grep -v '\.pt$' | grep -v README.md | wc -l
```

4. Quick sanity check of the numbers:

```python
import json; r = json.load(open("<folder>/results.json"))
for m, x in r.items():
    s = x.get("pair", x)                       # multitask nests under "pair"
    print(m, "best epoch", x["best_epoch"], "| seen", round(s["seen"]["1"], 1),
          "| unseen", round(s["unseen"]["1"], 1), "| H", round(s["harmonic"]["1"], 1))
```

✅ The file count matches; every model has a best epoch; seen > unseen (expected for all tracks); nothing is exactly 0 or 100.

---

## Phase 6: write your run README (30–45 minutes)

Copy your track's `README_TEMPLATE.md` into the results folder as `README.md` and fill in **every** section:
- **Environment:** GPU, versions, Kaggle dataset version, repo commit.
- **Exact commands**, in order.
- **Pipeline**, plus any change from the protocol (normally none).
- **Process log:** everything you ran, including the smoke test, crashes and abandoned runs, with what you decided each time.
- **Training behaviour:** best epoch, time per epoch, the train vs val gap, how `val_unseen` behaved.
- **Results**, copied from `summary.csv`, plus a v1 comparison using "seen excl. held-out".
- **Error analysis** from the unseen confusion matrices.
- **Checkpoints:** file, size, SHA-256, epoch, val H, date sent (fill the date in Phase 8).

Get the SHA-256 values in Kaggle with `!sha256sum /kaggle/working/<folder>/*.pt`, or locally with `shasum -a 256 *.pt`.

✅ No `<placeholder>` text is left in your README.

---

## Phase 7: commit and push (10 minutes)

```bash
git pull
git switch -c v2/<track>                    # v2/crop, v2/disease or v2/multitask
git add <your results folder> <your notebook, crop/disease only> <changed src/ files, multitask only>
git status                                  # check: no .pt files listed (they are git-ignored)
git commit -m "Add v2 <track> results (Kaggle dataset v<n>)"
git push -u origin v2/<track>
```

Then open a pull request into `master` (or merge, whichever the team agreed).

✅ `git status` is clean, the branch is on GitHub, and no `.pt` file was committed.

---

## Phase 8: tell the group, then send the checkpoints

1. **Post the "done" message** in the WhatsApp group (your track's exact template is in your `README.md`):
   ```
   [v2 <track>] DONE
   What: <what you trained>
   Best (Top-1): <model> – seen xx.x / unseen xx.x / H xx.x
   Pushed: <results folder> (commit <hash>)
   Checkpoints: <file names> (sent below; SHA-256 in README)
   Notes: <anything off-protocol, problems, or "none">
   ```
2. **Right after it, send the `.pt` files** to the group: attach → **Document** (not Gallery / Photo / Video), so the file arrives unchanged. Up to 2 GB per file.
3. Fill the "date sent" column in your README's checkpoint table and push that small change.

✅ The group has your message, followed by every checkpoint file.

---

## Phase 9: combine (whoever writes the results section)

When all three "DONE" messages are in:
- [ ] All three used the **same Kaggle version** and started from the same commit (check the STARTING messages and each README).
- [ ] Each `results.json` has `val`, `val_unseen`, `seen`, `seen_excl_heldout`, `unseen`, `harmonic` (multitask: under `pair`, `crop_head`, `disease_head`).
- [ ] Build one table: crop, disease, multitask (frozen and fine-tuned), with seen, unseen, H Top-1/Top-5 and per-class Top-1.
- [ ] Anyone who wants a checkpoint downloads it from the group and checks it with `shasum -a 256` against the README.

---

## If something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `IndexError` on the `glob(... label_maps.json ...)` line | dataset not attached to the notebook | Add Input → the team dataset, version <n> |
| Split sizes differ from 6995 / 773 / 227 / 2000 / 1448 | wrong dataset version (v1 attached) | attach v2; post in the group if unsure |
| Error downloading pretrained weights (timm / torchvision / Hugging Face) | Internet off | Settings → Internet on |
| `cuda False` | no GPU selected | Settings → Accelerator → GPU |
| `CUDA out of memory` | GPU too small or shared | add `--batch-size 32`; write it in your README as a deviation and say so in the DONE message |
| DataLoader worker errors or freezing | too many workers | `--workers 2` (or `0` to debug) |
| Session stops after closing the browser | interactive session timed out | run with **Save Version → Save & Run All** (Phase 4) |
| Run stopped half-way | Kaggle time or quota limit | crop/disease: finished backbones are in `results.json`; rerun only the missing ones with `--models <name>` into the same `--out` folder, then post in the group |
| `git apply` fails (multitask) | repo not at the right commit, or patch already applied | `git pull`; `git apply --check -R multitask_cnn/v2_protocol.patch` tells you if it is already applied |
| `.pt` shows up in `git status` | file outside the ignored patterns | don't add it; checkpoints go to WhatsApp only |
| Numbers look wrong (0%, 100%, unseen > seen) | something is off | stop, don't push; post the summary table in the group |
