# Multitask CNN: v2 retrain

**You:** retrain the multitask CNN (ConvNeXt-Tiny with a crop head and a disease head, scored on crop+disease pairs) on the new **DPD_Small_clean_v2** dataset: one run with a frozen backbone, one fine-tuned. Then hand in the results.
**Time:** about 3 h of GPU for both runs (Kaggle T4), plus about 1 h of your own time.
**Using an AI agent?** Give it [`HANDOFF_v2.md`](HANDOFF_v2.md). It has every command, file name and check in detail. This page is the short version for you.
**Start here:** [`V2_WORKFLOW.md`](../V2_WORKFLOW.md) is the end-to-end, step-by-step workflow all three tracks follow, with sync points in the WhatsApp group and what you should see at each step.
**Data preparation:** don't run any. The Kaggle dataset is already built and verified. What each data-preparation script did, for the report, is in the dataset's `V2_CHANGES.md` section 7.

v1 instructions for the baseline are still in [`RUN_BASELINE.md`](RUN_BASELINE.md).

## Why we are redoing it

Your v1 pipeline was the one that followed the team rules. v2 keeps your code and makes three changes:
- adds a validation set of unseen crop+disease pairs, and picks the best epoch on the harmonic mean of seen and unseen validation accuracy;
- adds early stopping;
- adds a fine-tuned run, so you can be compared with the two independent tracks, which fine-tune their backbones.

The code changes are already prepared and tested.

## Steps

1. **Read** the dataset's `README.md`: the "Quick start" and "Shared protocol" sections, about 5 minutes.
2. **Apply the prepared changes** from the repo root, then commit the changed `src/` files:
   ```
   git apply multitask_cnn/v2_protocol.patch
   ```
3. **Put the dataset** at `multitask_cnn/dataset/DPD_Small_clean_v2/`. On Kaggle, link the attached dataset there; [`HANDOFF_v2.md`](HANDOFF_v2.md) section 3 has the exact cell.
4. **Run**, from `multitask_cnn/`:
   ```
   python -m src.train_multitask --epochs 30 --run convnext_mt_frozen_v2
   python -m src.evaluate_v2 --run convnext_mt_frozen_v2 --name ConvNeXt-Tiny-frozen --out output/multitask_v2

   python -m src.train_multitask --epochs 30 --finetune --run convnext_mt_finetune_v2
   python -m src.evaluate_v2 --run convnext_mt_finetune_v2 --name ConvNeXt-Tiny-finetune --out output/multitask_v2
   ```
5. **Copy** each run's `train_log.csv` and `config.json` into `output/multitask_v2/` as `<name>_history.csv` and `<name>_config.json`, and save a curves plot per run (`<name>_curves.png`).
6. **Write the run README:** copy [`README_TEMPLATE.md`](README_TEMPLATE.md) into `output/multitask_v2/` as `README.md` and fill in every section. Include how you trained, everything you tried (failed runs too), the results, and the checkpoint list.
7. **Push** to the repo.
8. **WhatsApp:** post in the team group that you're done and what you did (format below), then send the 2 `.pt` files there as **Document** (not as photo or video). Rename each `runs/<run>/best.pt` to `<name>_best.pt` first.

## WhatsApp message

```
[v2 multitask] DONE
What: multitask CNN, ConvNeXt-Tiny frozen + fine-tuned, shared v2 protocol
Best (Top-1): pair, <frozen | finetune> – seen xx.x / unseen xx.x / H xx.x
Pushed: multitask_cnn/output/multitask_v2/ (commit <hash>)
Checkpoints: ConvNeXt-Tiny-frozen_best.pt, ConvNeXt-Tiny-finetune_best.pt (sent below; SHA-256 in README)
Notes: <anything off-protocol, problems, or "none">
```

The numbers come from `summary.csv`. The `.pt` files are not in git: they are too big for GitHub and ignored by `.gitignore`.

## What you hand in

| Where | What |
|---|---|
| `multitask_cnn/src/` | the patched `common.py` and `train_multitask.py`, plus `evaluate_v2.py` |
| `multitask_cnn/output/multitask_v2/` | everything `evaluate_v2` wrote, histories, configs, curves, your filled `README.md` |
| Team WhatsApp group | the "done" message + `ConvNeXt-Tiny-frozen_best.pt`, `ConvNeXt-Tiny-finetune_best.pt` |

## Rules

- **The two runs above use the settings as given.** Want to try your own ideas (loss weighting, class weights)? Run them afterwards into a separately named folder, each with its own README.
- **Never** use `test_seen` or `test_unseen` to choose epochs, learning rates or models.
- **Don't edit or delete** the v1 results in `output/convnext_mt_baseline_30ep/`.
- **Stuck or seeing something odd?** Ask in the WhatsApp group before changing the protocol.

## More detail

- [`HANDOFF_v2.md`](HANDOFF_v2.md): the full handoff (for you or your agent). Covers the Kaggle setup, the `results.json` format, checks before pushing, and what to look at for the report.
- Dataset `README.md` and `V2_CHANGES.md` (inside the dataset): what v2 contains and how it differs from v1 and the original.
