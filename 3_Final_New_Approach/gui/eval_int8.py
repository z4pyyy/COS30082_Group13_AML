"""Run the int8 browser models over every evaluation split, so the GUI reports what the browser computes.

    python3.14 3_Final_New_Approach/gui/eval_int8.py

Saves 3_Final_New_Approach/training/results/runs/gui_crop_int8/ and gui_dis_attn_int8/ logits.npz (same format as train.py),
then prints top-1 agreement and accuracy vs the original (bf16 GPU) logits.
Next: python3.14 3_Final_New_Approach/gui/build_assets.py --crop gui_crop_int8 --dis gui_dis_attn_int8
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent            # 3_Final_New_Approach/gui
ROOT = HERE.parents[1]                             # repo root
sys.path.insert(0, str(HERE.parent / "training"))
from common import DATA, EV, RESULTS, Labels, load_logits, save_logits  # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
M = HERE / "static" / "models"


def x_of(path):   # plain resize 224x224 + ImageNet normalise, as in app.js
    im = Image.open(DATA / path).convert("RGB").resize((224, 224), Image.BILINEAR)
    return ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None].copy()


def main():
    L = Labels()
    so = ort.SessionOptions()
    for name, onnx, task in (("gui_crop", "dino_cps_crop", "crop"), ("gui_dis_attn", "dino_cps_disease", "disease")):
        s = ort.InferenceSession(str(M / f"{onnx}.int8.onnx"), so, providers=["CPUExecutionProvider"])
        out = {sp: torch.from_numpy(np.concatenate([s.run(None, {"input": x_of(p)})[0] for p in L.dfs[sp].path])) for sp in EV}
        save_logits(RESULTS / "runs" / f"{name}_int8" / "logits.npz", out)
        ref = load_logits(RESULTS / "runs" / name / "logits.npz")
        for sp in EV:
            y = torch.tensor(L.dfs[sp][f"{task}_id"].values)
            agree = (out[sp].argmax(1) == ref[sp].argmax(1)).float().mean().item() * 100
            acc_i, acc_r = ((v.argmax(1) == y).float().mean().item() * 100 for v in (out[sp], ref[sp]))
            print(f"{name:13s} {sp:11s} top1 agree {agree:5.1f}%  acc int8 {acc_i:5.2f}  original {acc_r:5.2f}", flush=True)


if __name__ == "__main__":
    main()
