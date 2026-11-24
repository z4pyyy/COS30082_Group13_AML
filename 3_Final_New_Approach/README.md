# Final New Approach: DINO-CPS

**DINO-CPS = DINOv2 Compositional Pair Scoring.** Two fully fine-tuned DINOv2 ViT-L/14 models, one for the crop and one for
the disease, combined by a fixed rule over all 333 crop-disease pairs. There is no learned pair network, so an unseen pair
is scored exactly like a seen one.

```
leaf image ─┬─ crop model    (DINOv2 ViT-L, CLS token → 55 crops)          ─┐
            └─ disease model (DINOv2 ViT-L, attention pooling → 175 diseases) ─┤
                                                                             ▼
   score(c, d) = log p(c) + log p(d) + w·CNN(c, d) − b·seen(c, d) + q·host_prior(c, d)  →  rank 333 pairs
```

- `w · CNN(c, d)`: the team ConvNeXt-T baseline pair score (`../2_Baselines`).
- `b`: removes the bias towards the 317 pairs seen in training (calibrated stacking).
- `q · host_prior`: +1 if the disease occurs on a crop of the same botanical family, −1 for narrow-host diseases.
- `w`, `b`, `q`, epochs and the final variant are chosen on `val` + `val_unseen` only. Test is scored once.

## Folders

| Folder | What it is |
|---|---|
| [`training/`](training/) | Training, evaluation and the ablation study (3 seeds). Results in `training/results/` (`REPORT.md`, `summary.csv`, confusion matrices). |
| [`gui/`](gui/) | In-browser GUI: int8 ONNX export, int8 evaluation, asset build. `gui/static/` is the Hugging Face Space root. |

## Results (test, mean of 3 seeds)

| | Seen T1 | Unseen T1 | H T1 | H T5 |
|---|---|---|---|---|
| Best baseline (Indep. ConvNeXt-T + calibration) | 54.3 | 56.6 | 55.4 | 82.9 |
| **DINO-CPS** | **63.8 ± 1.7** | **76.1 ± 1.8** | **69.4 ± 1.3** | **93.1 ± 0.4** |

Honest limit: unseen Top-1 mostly measures the crop model. Only 16 pairs have no training images, on 13 crops, and 10 of those crops
have exactly one such pair, so a correct crop plus the calibration bias gives the correct pair. The disease model alone is right on only 8.0% of unseen-pair images
(`disease acc test_unseen` in `training/results/heads.csv`).

## GUI: rebuild and deploy

The GUI runs the int8 models in the browser (ONNX Runtime Web); no image is uploaded. Its numbers come from the int8 models
(H T1 66.9), not the 3-seed mean.

```
python3.14 3_Final_New_Approach/gui/export_onnx.py --weights 3_Final_New_Approach/training/weights   # gui_crop.pt, gui_dis_attn.pt → int8 ONNX
python3.14 3_Final_New_Approach/gui/eval_int8.py
python3.14 3_Final_New_Approach/gui/build_assets.py --crop gui_crop_int8 --dis gui_dis_attn_int8
# models: HF model repo (Spaces have a 1 GB storage cap); after a new upload, bump MODEL_CACHE in gui/static/app.js
hf upload z4pyyy/dino-cps 3_Final_New_Approach/gui/static/models . --repo-type model
# page: HF Space, without the models
hf upload z4pyyy/PlantDiseaseValidator 3_Final_New_Approach/gui/static . --repo-type=space --exclude "models/*"
```

The page loads the models from https://huggingface.co/z4pyyy/dino-cps.
