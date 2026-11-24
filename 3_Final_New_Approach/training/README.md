# DINO-CPS training: make the disease model learn the disease

DINO-CPS (DINOv2 Compositional Pair Scoring) is our final approach. Two fully fine-tuned DINOv2 ViT-L models
(crop; disease with attention pooling) are combined by a fixed rule over all 333 crop-disease pairs.

Goal: break the "crop → usual disease" shortcut found in an earlier frozen-feature prototype (its disease head
alone was right on only 4.6% of unseen-pair images). Each run adds one component, so the results form an
ablation study. The final pick is `dis_attn` (attention pooling), chosen on validation H only.

| Run | Adds | Script |
|---|---|---|
| `dis_base` | DINOv2 ViT-L fully fine-tuned, CLS head (reference) | `train.py` |
| `dis_attn` | R1: attention pooling over patch tokens (weak lesion finder) | `train.py --attn` |
| `dis_attn_adv` | R2: gradient-reversal crop adversary (crop-blind disease feature) | `--adv 0.3` |
| `dis_attn_adv_syn` | R4: 4,000 cut-paste images: lesions moved onto new host crops | `make_synthetic.py`, `--synth` |
| `dis_full` | R6: field-style augmentation, field oversampling, lab/field adversary | `--field` |
| `vl`, `vl_zeroshot` | R3: BioCLIP-2 matched to crop-free symptom text (`symptoms.json`) | `vl_finetune.py` |
| (in evaluation) | R5: host-range prior from botanical family (`common.py: FAMILY`) | `evaluate.py` |
| `crop` | Crop model, fine-tuned with field augmentation, shared by all rows | `train.py --task crop` |

Protocol (same as the baselines): 333 pairs ranked, every choice (epoch, bias, prior weight, fusion weight) is
tuned on `val` + `val_unseen`, test is scored once. Train data is DPD-Small only.

## Run (we used an RTX 5090)

1. Get the code and the dataset.
   ```
   git clone <repo> && cd COS30082_Group13_AML
   ```
   Copy the `DPD_Small_clean/` folder (302 MB, not in git; built by `1_Data_Preparation/`) into the repo root.

2. Make a Python 3.11 or 3.12 environment and install the CUDA build of torch first.
   ```
   python -m venv .venv
   .venv\Scripts\activate            (Linux: source .venv/bin/activate)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r 3_Final_New_Approach/training/requirements.txt
   ```

3. Check the pipeline (about 3 minutes, tiny model, writes `results_smoke/`, which git ignores).
   ```
   python 3_Final_New_Approach/training/run_all.py --smoke
   ```

4. Run everything (3 seeds, about 4–5 hours). Model weights download from Hugging Face on the first run.
   ```
   python 3_Final_New_Approach/training/run_all.py
   ```
   If it stops, run the same command again. Finished runs are skipped.
   Short on time: `python 3_Final_New_Approach/training/run_all.py --seeds 0` (about 1.5 hours).
   Windows DataLoader errors: add `--workers 4` (or `--workers 0`).

## Outputs (`3_Final_New_Approach/training/results/`)

| File | Content |
|---|---|
| `REPORT.md` | Mean ± sd over seeds for every model × variant, sorted by val H; the final pick |
| `summary.csv`, `summary_all.csv` | Same, as CSV (per seed in `_all`) |
| `heads.csv` | Crop / disease accuracy of each run alone, per split |
| `final_cm/` | Confusion matrices (seen and unseen test) of the final pick: brief requirement |
| `runs/<name>/` | `logits.npz` (log-probs), `meta.json`, `train_log.csv`, `attention.png` (R1 runs) |
| `synthetic_preview.png` | 8 examples of the cut-paste images |

The metric to watch: `disease acc test_unseen` (disease model alone on unseen-pair images).
It shows whether the model learned the disease rather than the crop shortcut.
