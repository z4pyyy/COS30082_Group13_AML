# v3_disease: make the disease model learn the disease

Goal: break the "crop → usual disease" shortcut found in `vlm_v2` (the disease head alone was right on only
4.6% of unseen-pair images). Each run adds one recommendation, so the results form an ablation ladder.

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

Protocol is unchanged from v2: 333 pairs ranked, every choice (epoch, bias, prior weight, fusion weight) is
tuned on `val` + `val_unseen`, test is scored once. Train data is DPD-Small only.

## Run on the RTX 5090 PC

1. Get the code and the dataset.
   ```
   git clone <repo> && cd COS30082_Group13_AML
   git checkout v3/disease
   ```
   Copy the `DPD_Small_clean_v2/` folder (302 MB, not in git) into the repo root, next to `v3_disease/`.

2. Make a Python 3.11 or 3.12 environment and install the CUDA build of torch first.
   ```
   python -m venv .venv
   .venv\Scripts\activate            (Linux: source .venv/bin/activate)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r v3_disease/requirements.txt
   ```

3. Check the pipeline (about 3 minutes, tiny model, writes `results_smoke/`, which git ignores).
   ```
   python v3_disease/run_all.py --smoke
   ```

4. Run everything (3 seeds, about 4–5 hours). Model weights download from Hugging Face on the first run.
   ```
   python v3_disease/run_all.py
   ```
   If it stops, run the same command again. Finished runs are skipped.
   Short on time: `python v3_disease/run_all.py --seeds 0` (about 1.5 hours).
   Windows DataLoader errors: add `--workers 4` (or `--workers 0`).

5. Commit and push the results.
   ```
   git add v3_disease/results
   git commit -m "v3_disease results"
   git push
   ```

## Outputs (`v3_disease/results/`)

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
