<!-- Copy this file to output/<your_run_folder>/README.md and fill in every section.
     Delete these comments and the example text in angle brackets. Someone else should be able to
     reproduce your numbers from this file alone. -->

# <Track name> on DPD_Small_clean_v2: <run folder name>

**Author:** <name>  ·  **Date:** <YYYY-MM-DD>  ·  **Track:** <independent crop CNN | independent disease CNN | multitask CNN>
**Run type:** <main protocol run | ablation: what it changes>

## 1. Summary

<3–5 lines: what you trained, the headline numbers (seen, unseen, harmonic mean Top-1 for the best model), and the one thing a reader should take away.>

## 2. Environment

| Item | Value |
|---|---|
| Platform | <Kaggle / Colab / local> |
| GPU | <e.g. Tesla T4 16 GB> |
| Python / torch / torchvision / timm | <versions> |
| Dataset | DPD_Small_clean_v2, Kaggle version <n> |
| Repo commit | <git hash of the code you ran> |
| Notebook | <file name of the notebook you committed, with outputs> |

## 3. Exactly how to reproduce

```
<the exact commands / notebook cells you ran, in order, including any flags>
```

## 4. Pipeline

- **Data:** `get_loaders()` from the dataset, splits train 6,995 / val 773 / val_unseen 227 / test_seen 2,000 / test_unseen 1,448. <Anything different? Say so.>
- **Augmentation:** the dataset loader's train augmentation. <Anything added?>
- **Sampler / class weights:** <none | which>
- **Model(s):** <backbones with exact pretrained weights, e.g. timm convnext_tiny.fb_in1k; heads>
- **Loss:** <e.g. CrossEntropy, label smoothing 0.1>
- **Optimiser and schedule:** <AdamW lr …, wd 0.05, OneCycle / cosine>
- **Epochs / patience / batch / seed:** <30 / 10 / 64 / 42>
- **Model selection:** best epoch = highest H(val Top-1, val_unseen Top-1). <Anything different?>
- **Changes from the shared protocol:** <none | list each change and why>

## 5. Process log

<Everything you did, in order, including what did not work. For each attempt: what you changed, why, what happened, and what you decided. Crashed or abandoned runs belong here too.>

| # | Date | What was run | Result | Decision |
|---|---|---|---|---|
| 1 | | | | |

## 6. Training behaviour

<Per backbone: best epoch, epochs run, time per epoch and total time, the train vs val gap, how val_unseen behaved during training. Embed or link <Model>_curves.png.>

| Model | Best epoch | Epochs run | Time / epoch | Total time | Clean train Top-1 | Val Top-1 | Val-unseen Top-1 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 7. Results (test sets, used once)

Copy from `summary.csv`. All values in %.

| Model | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 | Seen excl. held-out Top-1 | Seen per-class | Unseen per-class |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

<Multitask: also the crop-head and disease-head accuracy, seen and unseen.>

**Compared with v1** (use "seen excl. held-out" for seen): <table or 2–3 lines.>

## 8. Error analysis

<From the confusion matrices and prediction files: which classes / pairs fail on test_unseen, what they are confused with, and why you think so. Link the relevant PNGs.>

## 9. Checkpoints

Not in git (`.gitignore` excludes `*.pt`; GitHub rejects files over 100 MB). Sent to the team WhatsApp group as documents, right after the "done" message.

| File | Size | SHA-256 | Epoch | Val H | Sent to WhatsApp group (date) |
|---|---|---|---|---|---|
| <Model>_best.pt | | | | | |

How to load:

```python
<the 3–4 lines that rebuild the model and load the state dict>
```

## 10. Known issues and open questions

<Anything odd, suspected bugs, numbers you do not trust, things you would try next.>
