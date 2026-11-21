# New approach (zero-shot) on DPD_Small_clean_v2: zsl_independent_v2

**Author.** Samuel Chang Hao Jun (102775957)  ·  **Date.** 2026-11-20  ·  **Track.** new approach (compositional zero-shot learning)
**Run type.** final run `zsl_final`, frozen independent models + OADis-inspired zero-shot learning head (ZSLHead) + calibrated stacking + late fusion (ZSLFusion)

## 1. Summary

We kept the team's independent structure (the crop ConvNeXt-Tiny and the disease ConvNeXt-Tiny from the crop and disease tracks, both frozen) and added a compositional zero-shot learning (CZSL) step on top of it. The approach has three parts.

1. **OADis-inspired zero-shot learning head (ZSLHead).** A small network on top of the frozen crop and disease models. It learns how a crop and a disease combine. During training it also mixes the crop feature of one image with the disease feature of another, to make training features for the 16 pairs that have no training images (the 10 unseen test pairs and the 6 held-out validation pairs). Only the pair names are used for this, never their images. The idea of hallucinating unseen compositions comes from OADis (Saini et al., CVPR 2022). We did not use the OADis disentangling module, because our two independent models already give separate crop and disease features.
2. **Calibrated stacking** (Chao et al., ECCV 2016). A bias subtracted from the scores of pairs seen in training, so unseen pairs are not always outscored.
3. **Late fusion (ZSLFusion).** ZSLFusion is the late fusion of the team's product score with the ZSLHead score. The ZSLHead score is added to the product score with a weight w, instead of replacing it. ZSLFusion followed by calibrated stacking is called ZSLFusion-calib, our final model.

The fusion weight, the calibration bias and ZSLHead's best epoch were all chosen on `val` and `val_unseen` only. The test sets were scored once, after everything was fixed.

**Main result (pair level, Top-1, %).**

| Row | Seen | Unseen | H | AUC |
|---|---|---|---|---|
| Product (team baseline, same pipeline) | 63.80000114440918 | 13.604971766471863 | 22.42742763491214 | 35.276156585624584 |
| **ZSLFusion-calib (our final model)** | 53.75000238418579 | 58.08011293411255 | **55.831225780226305** | **36.66068750958964** |

Take-away. Against the team baseline, H goes from 22.43 to 55.83 and unseen accuracy from 13.60 to 58.08. Most of that gain comes from calibrated stacking, since calibration alone on the baseline already reaches H 55.38. ZSLHead adds a small further gain on top (H +0.44993210012514595, AUC +1.384530923965059), and validation chose a fusion weight of 0.5 rather than 0, so ZSLHead does carry some information the independent models lack. The gain is small, and we think the main limit is that the disease features still come from the team's disease model, which learned the crop shortcut.

## 2. Environment

| Item | Value |
|---|---|
| Platform | local Windows PC (not Colab), VS Code with Jupyter |
| GPU | NVIDIA GeForce RTX 4060 |
| Python / torch / timm | 3.13.15 / 2.11.0+cu128 / 1.0.31 |
| Dataset | DPD_Small_clean_v2, Kaggle version 3 (team dataset AML_Group_Assignment, folder DPD_Small_clean_v2) |
| Repo commit | `419fdeb` (master, full hash 419fdeb6ed5760a745cb2c56e28495263ce9789a) |
| Notebook | `zsl_local.ipynb` (committed with outputs) |
| Data loader workers | 0 |

The crop and disease tracks ran on a Colab T4 with torch 2.11.0+cu130 and timm 1.0.30. Our torch and timm versions are the same or one patch apart, with a different CUDA build. Because both backbones are frozen and loaded from the teams' checkpoints, this does not change the comparison. Our recomputed baseline (H 22.42742763491214) matches the team's pair result (H 22.42) and the disease track's shortcut figure (61.89710610932476% against 61.8%).

## 3. Exactly how to reproduce

Folder layout on the PC

```
D:\DPD\
├── zsl_local.ipynb
├── DPD_Small_clean_v2\                (v2 dataset from Kaggle, unzipped, unchanged)
├── checkpoints\
│   ├── ConvNeXt-Tiny_crop.pt          (crop track ConvNeXt-Tiny_best.pt, SHA-256 2980a866…)
│   └── ConvNeXt-Tiny_disease.pt       (disease track ConvNeXt-Tiny_best.pt, SHA-256 c18c6bbd…)
└── zsl-env\                           (Python 3.13 venv)
```

Setup

```
python -m venv zsl-env
zsl-env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm pandas matplotlib ipykernel
```

Then open `zsl_local.ipynb` in VS Code with the `zsl-env` kernel and run it from the top.

- Sections 1 to 3 hold the settings and code, and check the GPU, the split sizes (6995 / 773 / 227 / 2000 / 1448), that all 11,443 images exist, and both checkpoints' SHA-256.
- Section 4 is a short pipeline test.
- Section 5 is the main run into `output\zsl_final`. One pipeline trains ZSLHead, tunes the calibration biases and the ZSLFusion weight on validation, then scores the test sets once (feature extraction about 3 minutes the first time, ZSLHead training about 1 minute on the RTX 4060).
- The run uses PyTorch's deterministic mode (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic`, `use_deterministic_algorithms`), so a rerun gives the same numbers.

## 4. Pipeline

- **Data.** The dataset's own loader (`get_loaders()`, `DPDDataset`, `build_transforms`), unchanged. Same splits as every track.
- **Backbones.** The crop track's and disease track's `ConvNeXt-Tiny_best.pt` (`convnext_tiny.fb_in1k`), loaded unchanged and frozen. They never share weights, so the independent structure is kept.
- **Features.** For each image, the 768-d pooled feature and the logits of both models. `train` is passed once without augmentation plus twice with the loader's own train augmentation (20,985 training rows). Features are cached once in `zsl_features`.
- **Pair embedder.** Each of the 55 crops and 175 diseases has a 300-d name vector, learned from scratch. A pair's embedding is a two-layer MLP of the joined [crop vector, disease vector], for all 333 pairs.
- **Composer.** A two-layer MLP of the joined [crop feature, disease feature] (each LayerNorm'd) into the same 512-d space.
- **Scoring.** Cosine similarity between the image vector and every pair vector, divided by tau 0.05, over all 333 pairs.
- **Hallucination (zero-shot part).** Every batch also builds fake features for target pairs (c, d) by taking the crop feature from a training image of crop c and the disease feature from a training image of disease d. Half the targets are the 16 pairs with no training images.
- **Loss.** Cross-entropy with label smoothing 0.1 on real features + 1.0 × the same loss on hallucinated features.
- **Optimiser and schedule.** AdamW, weight decay 0.05, OneCycleLR (pct_start 0.1), lr 1e-3 for ZSLHead.
- **Epochs / patience / batch / seed.** 30 / 10 / 64 / 42. Early stop at epoch 27, best epoch 17.
- **Model selection.** Best epoch = highest H(val pair Top-1, val_unseen pair Top-1) of ZSLHead's own scores (val H 35.3314255568887 at epoch 17).
- **Calibrated stacking.** A bias subtracted from the scores of the pairs that have training images, picked by the highest H on `val` + `val_unseen`.
- **Late fusion (ZSLFusion).** ZSLFusion score = Product score + w × ZSLHead score. For each w in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0} the calibration bias was tuned on validation, and the (w, bias) with the highest validation H was kept. w = 0 is the same as Product-calib, so it stayed in the grid as the reference.
- **Changes from the shared protocol.**
  - No backbone is trained. Both are the team's checkpoints, frozen.
  - ZSLHead has its own learning rate (1e-3), because it trains on cached features.
  - Training features include two extra passes with the loader's train augmentation.
  - Hallucinated features for pairs without training images are made from the pair names only.

## 5. Process log

| # | Date | What was run | Result | Decision |
|---|---|---|---|---|
| 1 | 2026-11-20 | Local setup. Python 3.13 venv, torch 2.11.0+cu128, checkpoints and dataset checks | CUDA available on the RTX 4060, split sizes and all 11,443 images correct, both SHA-256 match the crop and disease READMEs | Continue |
| 2 | 2026-11-20 | Pipeline test (1 epoch, 3 batches) | Ran end to end, files written and deleted | Start the main run |
| 3 | 2026-11-20 | Main run `step2_comp_head` | Baseline reproduced (H 22.42742763491214). ZSLHead-calib H 42.87546717977811, but Product-calib reaches H 55.38129368010116. ZSLHead reaches 99.9% train accuracy, so it overfits the training features | Test whether ZSLHead adds anything on top of the baseline |
| 4 | 2026-11-20 | Late fusion `step2_fusion`, tuned on validation only, ZSLHead from run 3 not retrained | Best w = 0.5, bias 9.369862570762635, val H 68.9601622560166 (against 64.25953716789945 at w = 0). ZSLFusion-calib H 55.831225780226305 | Add late fusion to the method |
| 5 | 2026-11-20 | Final run `zsl_final`. The notebook merged into one pipeline (ZSLHead, calibration, ZSLFusion), deterministic mode on, run once | Identical to runs 3 and 4 in every metric, the per-epoch history, the fusion grid and the shortcut figures (only the seconds per epoch differ). The earlier folders `step2_comp_head` and `step2_fusion` were removed | Rename the models |
| 6 | 2026-11-20 | Renamed CompHead to ZSLHead and Fusion to ZSLFusion, rerun with deterministic mode, run once from a fresh kernel (`jupyter nbconvert --execute`) | Identical results to run 5 in every metric, the per-epoch history, the fusion grid, the shortcut figures, all prediction and confusion-matrix files and the head weights. Only the names, the seconds per epoch and the checkpoint's SHA-256 differ (the checkpoint stores the new name) | Final result |

A short ZSLHead retraining grid (dropout, hidden size, hallucination weight) was started before run 4 and stopped early. It never scored the test sets and its files were removed. We kept the ZSLHead settings from run 3.

## 6. Results (pair level, test sets, %)

From `output/zsl_final/summary.csv`

| Model | Val Top-1 | Val-unseen Top-1 | Seen Top-1 | Unseen Top-1 | H Top-1 | H Top-3 | H Top-5 | Seen excl. held-out Top-1 | Seen per-class | Unseen per-class |
|---|---|---|---|---|---|---|---|---|---|---|
| Product | 81.37 | 16.74 | 63.80 | 13.60 | 22.43 | 46.98 | 54.00 | 65.52 | 39.19 | 13.85 |
| Product-calib | 70.50 | 59.03 | 54.25 | 56.56 | 55.38 | 75.19 | 82.91 | 53.97 | 27.18 | 60.89 |
| ZSLHead | 82.66 | 22.47 | 65.85 | 13.47 | 22.36 | 39.49 | 46.52 | 67.42 | 47.53 | 13.24 |
| ZSLHead-calib | 72.19 | 59.47 | 54.45 | 35.36 | 42.88 | 60.83 | 66.17 | 54.48 | 30.32 | 38.57 |
| ZSLFusion | 82.54 | 18.06 | 66.05 | 13.81 | 22.85 | 46.37 | 55.96 | 67.78 | 45.54 | 14.36 |
| **ZSLFusion-calib** | 70.12 | 67.84 | 53.75 | 58.08 | **55.83** | 74.59 | 81.94 | 53.30 | 28.26 | 62.17 |

Unrounded values are in `results.json`.

## 7. Comparison with the baseline

`Product` is the team's independent ConvNeXt pair result, recomputed in PyTorch through the same pipeline.

| | Product (baseline) | Product-calib | ZSLFusion-calib (final) |
|---|---|---|---|
| Seen Top-1 | 63.80000114440918 | 54.25000190734863 | 53.75000238418579 |
| Unseen Top-1 | 13.604971766471863 | 56.56077265739441 | 58.08011293411255 |
| H Top-1 | 22.42742763491214 | 55.38129368010116 | 55.831225780226305 |
| H Top-3 | 46.97729968593706 | 75.1899057745076 | 74.5866262353748 |
| H Top-5 | 54.002066043248945 | 82.90801449258896 | 81.93792326684174 |
| AUC (seen vs unseen curve) | 35.276156585624584 | 35.276156585624584 | 36.66068750958964 |
| Shortcut share on test_unseen | 74.00662251655629 | 38.75598086124402 | 37.56177924217463 |

- Against the baseline, ZSLFusion-calib gains +33.40379814531417 H Top-1, mostly from much higher unseen accuracy at the cost of about 10 points of seen accuracy.
- Against calibration alone, the gain is +0.44993210012514595 H Top-1 and +1.384530923965059 AUC. At Top-3 and Top-5, ZSLFusion-calib is slightly below Product-calib. We read this as a small improvement, not a large one.

## 8. Analysis

- **Calibration does most of the work.** The baseline is strongly biased towards seen pairs (val_unseen 16.74 against val 81.37). A seen-pair bias of 5.507096464019269, tuned on validation, moves the test H from 22.43 to 55.38. This shows how much of the baseline's failure is bias rather than missing knowledge.
- **ZSLHead alone does not help.** Without fusion, ZSLHead reaches 99.66 clean train accuracy and its AUC (25.157889656112363) is below the baseline's (35.276156585624584). The backbones were fine-tuned on these same training images, so their train features are cleaner than test features and ZSLHead overfits them.
- **Late fusion recovers a small gain.** Used as an extra signal on top of the product score, ZSLHead improves validation H at every w above 0 (64.26 at w = 0 up to 68.96 at w = 0.5), and on test it raises H by 0.45 and AUC by 1.38. AUC does not depend on the chosen bias, so it is the cleaner sign that ZSLHead helps.
- **Crop and disease inside the pair.** On `test_unseen`, ZSLFusion-calib gets the crop right 73.20442199707031% of the time but the disease only 58.08011293411255%, so the disease half still limits unseen accuracy.
- **Shortcut.** The share of wrong unseen answers that are a disease already seen on the true crop falls from 74.00662251655629% (baseline) to 37.56177924217463% (ZSLFusion-calib). The disease model alone has 61.89710610932476%.
- **Per unseen pair (ZSLFusion-calib, test_unseen Top-1).** Strong on cucumber downy mildew (100.0%), sugarcane rust (85.39325842696628%), cucumber powdery mildew (83.78378378378379%), pepper bell bacterial spot (78.63636363636364%) and mango anthracnose (77.89473684210526%). Weak on corn brown spot (8.571428571428571%), cherry aphis spp (18.055555555555554%) and tomato early blight (37.71186440677966%). Tomato early blight is a third of `test_unseen` (472 images), so it pulls the unseen average down the most.
- **Training curves.** ZSLHead reaches 99.92 train accuracy by epoch 27 while val stays near 82 to 84 and val_unseen moves between 15.86 and 22.47, which is the overfitting described above. The train loss sits above the val loss because it adds the hallucinated-pair loss on top of the real-feature loss.
- **Limits.** `val_unseen` is small (6 pairs, 227 images), so the chosen epoch, bias and w can move a few points. The validation gain from fusion (+4.70062508811715 H) was larger than the test gain (+0.44993210012514595 H). The disease features still come from the team's disease model, which learned the crop shortcut. Retraining the disease model with same-disease-on-different-crop pairs (the OADis disentangling idea) is the natural next step.

## 9. Files

`output/zsl_final/` (85 files, `ZSLHead_best.pt` not in git)

| File | What it is |
|---|---|
| `config.json` | every setting, both backbone SHA-256 values, epochs and val H, the late fusion rule, the deterministic settings, versions, changes from the protocol |
| `results.json` | per row, Top-1/3/5 and per-class for val, val_unseen, seen, unseen, harmonic, seen excl. held-out, clean train Top-1, crop and disease Top-1 from the pair, shortcut share, AUC |
| `summary.csv` | same columns as the other tracks, one row per variant (6 rows) |
| `ZSLHead_history.csv`, `ZSLHead_curves.png` | per-epoch loss, val, val_unseen and H |
| `<Row>_<split>_confusion_matrix.csv/.png` | for each row and split (val, val_unseen, seen, unseen) |
| `<Row>_predictions_<split>.csv` | per-image top-5 pairs, plus the crop and disease of the top-1 pair |
| `all_models_accuracy.png`, `harmonic_mean.png` | bar charts |
| `calibration.json` | chosen bias per row (and w for ZSLFusion) with the full validation sweep |
| `zsl_fusion_tuning_val.csv` | the ZSLFusion grid, validation numbers only |
| `shortcut.json` | shortcut share per row and for the disease model alone |
| `auc_curve.png` | seen vs unseen accuracy as the bias sweeps |
| `run_log.txt` | console log of the run |

## 10. Checkpoint

Not in git (`.gitignore` excludes `*.pt`). Sent to the team WhatsApp group as a document, right after the DONE message.

| File | Size | SHA-256 | Epoch | Val H | Sent to WhatsApp group (date) |
|---|---|---|---|---|---|
| ZSLHead_best.pt | 13,271,843 bytes | `8951bf7a557b1842bef5a99735684303fc6349baae7a8300d4aa0c6a8292cc4c` | 17 | 35.3314255568887 | not yet sent |

ZSLHead needs both backbone checkpoints to run (crop `2980a866…`, disease `c18c6bbd…`). ZSLFusion has no weights of its own, only w = 0.5 and bias 9.369862570762635. They are stored in `calibration.json`, `config.json` and inside `ZSLHead_best.pt`.

## 11. Code

| File | What it does |
|---|---|
| `zsl_local.ipynb` | the notebook that produced `output/zsl_final` in one run (ZSLHead, calibration, ZSLFusion, final test) |

## 12. References

- N. Saini, K. Pham and A. Shrivastava, "Disentangling Visual Embeddings for Attributes and Objects", CVPR 2022 (OADis).
- W.-L. Chao, S. Changpinyo, B. Gong and F. Sha, "An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild", ECCV 2016 (calibrated stacking).
- A. Chai, S. H. Lee et al., "Pairwise Feature Learning for Unseen Plant Disease Recognition", ICIP 2023.
- Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell and S. Xie, "A ConvNet for the 2020s", CVPR 2022 (ConvNeXt).
