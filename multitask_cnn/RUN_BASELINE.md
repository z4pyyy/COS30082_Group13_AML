# Multi-task CNN baseline (ConvNeXt-Tiny)

Brief section 3.1, Fig. 1 (right): one ConvNeXt-Tiny backbone, a crop head (55) and a disease head (175).
Plain baseline, nothing tuned: ImageNet weights, backbone frozen, only the two linear heads train,
unweighted CE(crop) + CE(disease), AdamW lr 1e-3, 15 epochs, batch 64, team data loader as is.

## Setup (once)

```
pip install -r requirements.txt
```

For a GPU, install the CUDA build of PyTorch from pytorch.org first.

## Train, then evaluate (run from this folder)

```
python -m src.train_multitask
python -m src.evaluate --run convnext_mt_baseline
```

The committed results are the 30-epoch run. To reproduce them:

```
python -m src.train_multitask --epochs 30 --run convnext_mt_baseline_30ep
python -m src.evaluate --run convnext_mt_baseline_30ep
```

Results go to `runs/convnext_mt_baseline/` (git-ignored; the committed 30-epoch run is in `output/convnext_mt_baseline_30ep/`): `metrics.json` (Top-1/Top-5 seen and unseen, harmonic mean,
crop/disease accuracy), `confusion_test_seen.png/.csv`, `top_confusions_test_seen.csv`, `train_log.csv`.

The backbone is picked by validation only. The test sets are used once, in `evaluate.py`.

## How unseen comps are predicted

score(comp) = log P(crop) + log P(disease), over all 333 comps, then top-1 / top-5.
Each unseen comp is a seen crop + seen disease that never appear together in training.

## Later, the fine-tuned version

```
python -m src.train_multitask --finetune --run convnext_mt_finetune
```
