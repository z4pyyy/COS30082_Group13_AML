# Independent crop CNN: v2 retrain

**You:** retrain the crop classifier (55 crops) with 3 backbones on the new **DPD_Small_clean_v2** dataset, then hand in the results.
**Time:** about 1.5 h of GPU (Kaggle T4), plus about 1 h of your own time.
**Using an AI agent?** Give it [`HANDOFF_v2.md`](HANDOFF_v2.md). It has every command, file name and check in detail. This page is the short version for you.
**Start here:** [`V2_WORKFLOW.md`](../V2_WORKFLOW.md) is the end-to-end, step-by-step workflow all three tracks follow, with sync points in the WhatsApp group and what you should see at each step.
**Data preparation:** don't run any. The Kaggle dataset is already built and verified. What each data-preparation script did, for the report, is in the dataset's `V2_CHANGES.md` section 7.

## Why we are redoing it

The v1 run picked its best epoch on the seen test set (a test leak) and used no augmentation. v2 fixes both, adds a validation set of unseen crop+disease pairs, and puts all three model tracks on the same settings, so the report can compare them in one table.

## Steps

1. **Read** the dataset's `README.md`: the "Quick start" and "Shared protocol" sections, about 5 minutes.
2. **Kaggle notebook:** add the dataset `DPD_Small_clean_v2` (team Kaggle, v2 version), turn on GPU T4 and Internet.
3. **Run** the ready-made script. You don't need to write training code:
   ```python
   !pip -q install timm
   import glob, os
   DATA = os.path.dirname(glob.glob("/kaggle/input/**/label_maps.json", recursive=True)[0])
   !python {DATA}/train_independent.py --task crop --out /kaggle/working/crop_independent_v2 --workers 4
   ```
4. **Download** the output folder and put it in the repo at `crop_indi_cnn/output/crop_independent_v2/`. Save your notebook, with its outputs, as `crop_indi_cnn/crop_independent_cnns_v2.ipynb`.
5. **Write the run README:** copy [`README_TEMPLATE.md`](README_TEMPLATE.md) into that output folder as `README.md` and fill in every section. Include how you trained, everything you tried (failed runs too), the results, and the checkpoint list.
6. **Push** to the repo.
7. **WhatsApp:** post in the team group that you're done and what you did (format below), then send the 3 `.pt` files there as **Document** (not as photo or video).

## WhatsApp message

```
[v2 crop_independent] DONE
What: independent crop CNN, 3 backbones, shared v2 protocol
Best (Top-1): <best backbone> – seen xx.x / unseen xx.x / H xx.x
Pushed: crop_indi_cnn/output/crop_independent_v2/ (commit <hash>)
Checkpoints: ResNet50_best.pt, EfficientNet-B0_best.pt, ConvNeXt-Tiny_best.pt (sent below; SHA-256 in README)
Notes: <anything off-protocol, problems, or "none">
```

The numbers come from `summary.csv`. The `.pt` files are not in git: they are too big for GitHub and ignored by `.gitignore`.

## What you hand in

| Where | What |
|---|---|
| `crop_indi_cnn/output/crop_independent_v2/` | everything the script wrote + your filled `README.md` |
| `crop_indi_cnn/crop_independent_cnns_v2.ipynb` | the notebook you ran, with outputs |
| Team WhatsApp group | the "done" message + `ResNet50_best.pt`, `EfficientNet-B0_best.pt`, `ConvNeXt-Tiny_best.pt` |

## Rules

- **Main run: use the script as is.** Want to try your own ideas? Run them afterwards into a separately named folder, e.g. `output/crop_independent_v2_<idea>/`, each with its own README.
- **Never** use `test_seen` or `test_unseen` to choose epochs, learning rates or models. The script already handles this.
- **Don't edit or delete** the v1 results in `output/crop_independent/`.
- **Stuck or seeing something odd?** Ask in the WhatsApp group before changing the protocol.

## More detail

- [`HANDOFF_v2.md`](HANDOFF_v2.md): the full handoff (for you or your agent). Covers Colab instructions, the `results.json` format, checks before pushing, and what to look at for the report.
- Dataset `README.md` and `V2_CHANGES.md` (inside the dataset): what v2 contains and how it differs from v1 and the original.
