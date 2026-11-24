# COS30082 Group 13: Out-of-Distribution Plant Disease Identification

Predict the **crop + disease pair** of a leaf photo on DPD-Small (55 crops, 175 diseases, 333 pairs).
The 10 pairs in `test_unseen` never appear in training, so a model must combine a known crop with a known disease in a new way.

**Final model: DINO-CPS** (DINOv2 Compositional Pair Scoring), in [`3_Final_New_Approach/`](3_Final_New_Approach/).
**Live demo (runs in your browser):** https://z4pyyy-plantdiseasevalidator.static.hf.space

## Repository layout

```
1_Data_Preparation/      audit, clean and split DPD-Small → DPD_Small_clean (not in git)
2_Baselines/             brief baselines: independent CNNs (crop, disease) and multitask CNN
   Independent_CNN_Crop/     notebook + results, 3 backbones
   Independent_CNN_Disease/  notebook + results, 3 backbones
   Multitask_CNN/            code + results, 3 backbones
   train_independent.py      shared training script for the independent CNNs
   export_onnx.py            exports the team ConvNeXt-T CNNs used by the GUI
3_Final_New_Approach/    DINO-CPS
   training/                 train, evaluate, ablation runs, results (3 seeds)
   gui/                      in-browser GUI (Hugging Face static Space)
Report/                  report source (build_report.js), figures, PDF
```

## Results (test sets, %)

Pair accuracy. H = harmonic mean of seen and unseen. All choices tuned on validation only; test scored once.

| Model | Seen T1 | Unseen T1 | H T1 | Seen T5 | Unseen T5 | H T5 |
|---|---|---|---|---|---|---|
| Independent ResNet50 | 63.9 | 8.4 | 14.9 | 84.7 | 40.9 | 55.1 |
| Independent EfficientNet-B0 | 64.0 | 7.9 | 14.0 | 85.2 | 37.8 | 52.3 |
| Independent ConvNeXt-Tiny | 63.8 | 13.6 | 22.4 | 83.8 | 39.9 | 54.0 |
| Multitask ResNet50 | 61.8 | 4.1 | 7.6 | 85.3 | 31.7 | 46.2 |
| Multitask EfficientNet-B0 | 61.0 | 1.4 | 2.7 | 83.3 | 22.3 | 35.2 |
| Multitask ConvNeXt-Tiny | 63.3 | 4.2 | 7.9 | 84.0 | 20.9 | 33.5 |
| Indep. ConvNeXt-Tiny + calibration | 54.3 | 56.6 | 55.4 | 79.4 | 86.7 | 82.9 |
| **DINO-CPS** (mean of 3 seeds) | **63.8 ± 1.7** | **76.1 ± 1.8** | **69.4 ± 1.3** | **89.2 ± 0.8** | **97.3 ± 0.2** | **93.1 ± 0.4** |

## DINO-CPS in one paragraph

Two DINOv2 ViT-L/14 models are fully fine-tuned on DPD-Small: a crop model (CLS token → 55 classes) and a disease
model (gated attention pooling over the 256 patch tokens → 175 classes). Each pair (c, d) gets the score
log p(c) + log p(d) + w · CNN(c, d) − b · seen(c, d) + q · host-prior(c, d). The CNN term is the team ConvNeXt-T
baseline; b removes the bias towards pairs seen in training; w, b, q are tuned on validation.
Details, ablation and run commands: [`3_Final_New_Approach/README.md`](3_Final_New_Approach/README.md).

## Reproduce

1. **Data:** put the raw DPD-Small in `DPD_Small/`, then run `1_Data_Preparation/01_audit.py`,
   `03_build_dataset.py`, `04_verify_dataset.py`. See [`1_Data_Preparation/CLEANING_NOTES.md`](1_Data_Preparation/CLEANING_NOTES.md).
2. **Baselines:** the notebooks in `2_Baselines/Independent_CNN_*` (Colab, T4) and `2_Baselines/Multitask_CNN/README.md`.
3. **DINO-CPS:** `python 3_Final_New_Approach/training/run_all.py` (GPU, about 4–5 h for 3 seeds).

## Team

COS30082 Applied Machine Learning, 2026 S2, Group 13: Enoch Eren Chua (104401704), Terence Chin Seng Wong (104404059),
Samuel Hao Jun Chang (102775957), Ji Yan Lim (104382537).
