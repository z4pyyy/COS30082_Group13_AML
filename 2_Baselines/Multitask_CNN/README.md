# Multitask CNN baseline

Brief section 3.1, Fig. 1 (right): one shared backbone with a crop head (55 classes) and a disease head (175 classes).
Loss = CE(crop) + CE(disease). The pair score for the 333 crop-disease pairs is log p(crop) + log p(disease).

The run uses **the same recipe as the independent CNNs** (`../train_independent.py`), so the only difference is one network against two:

| Setting | Value |
|---|---|
| Backbones | timm `resnet50.tv2_in1k`, `efficientnet_b0.ra_in1k`, `convnext_tiny.fb_in1k` (ImageNet-1k), fully fine-tuned |
| Optimiser | AdamW, weight decay 0.05, OneCycleLR (pct_start 0.1), mixed precision |
| Learning rate | ResNet-50 3e-4, EfficientNet-B0 5e-4, ConvNeXt-Tiny 1e-4 |
| Loss | CE(crop) + CE(disease), label smoothing 0.1 on both heads |
| Batch / epochs / patience / seed | 64 / 30 / 10 / 42 |
| Model selection | best epoch = highest H(val Top-1, val_unseen Top-1) on pairs; test sets scored once |

## Run (from this folder)

```
pip install -r requirements.txt
# dataset: put DPD_Small_clean/ in dataset/ (see ../../1_Data_Preparation)
python -m src.train_multitask_aligned                       # all 3 backbones (GPU)
python -m src.train_multitask_aligned --models ConvNeXt-Tiny --accum 2   # small GPU: micro-batch 32 x 2
python -m src.evaluate --run mt_aligned_ConvNeXt-Tiny --name ConvNeXt-Tiny-MT --out output/multitask_aligned
```

## Files

| Path | What it is |
|---|---|
| `src/train_multitask_aligned.py` | training script used for the results |
| `src/train_multitask.py` | shared training helpers (validation loop) |
| `src/evaluate.py` | writes the standard result files (same schema as the independent CNNs) |
| `src/models.py`, `src/metrics.py`, `src/common.py`, `src/plot_curves.py` | model, metrics, paths, curves |
| `output/multitask_aligned/` | `summary.csv`, `results.json`, predictions and confusion matrices per backbone and split |
| `output/models/` | checkpoints (`*_best.pt`, not in git) |

## Results

Per-backbone test results are in `output/multitask_aligned/summary.csv`; the comparison with all models is in the root `README.md` and report Section 5.
The shared backbone lets the disease head use crop features, so it transfers worse to unseen pairs than the independent CNNs.
