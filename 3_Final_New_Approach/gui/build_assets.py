"""Build the data files of the static Hugging Face Space, from the repo (run on the Mac, after the GUI runs exist).

    python3.14 3_Final_New_Approach/gui/build_assets.py                      # uses runs gui_crop + gui_dis_attn (the saved-weights retrain)
    python3.14 3_Final_New_Approach/gui/build_assets.py --crop crop_s0 --dis dis_attn_s0   # dry run with an old run (no weights)

Writes (all gitignored, uploaded to the Space):
    3_Final_New_Approach/gui/static/data/params.json  b, q, w, host prior, seen mask, pair map (used by app.js)
    3_Final_New_Approach/gui/static/models/*.onnx     team ConvNeXt-T crop + disease (CNN fusion); DINO-CPS models come from export_onnx.py
    3_Final_New_Approach/gui/static/data/meta.json  names, pair map, examples, gallery index
    3_Final_New_Approach/gui/static/data/eval.json  brief metrics + sparse confusion matrices of the deployed model
    3_Final_New_Approach/gui/static/examples/       seen / unseen test images (copied from cnn_assets/)
    3_Final_New_Approach/gui/static/gallery/        up to 4 training images per seen pair (brief: "display the relevant training images")

The b, q, w search is the same as 3_Final_New_Approach/training/evaluate.py (+prior+CNN): tuned on val + val_unseen, test scored once.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent            # 3_Final_New_Approach/gui
ROOT = HERE.parents[1]                             # repo root
sys.path.insert(0, str(HERE.parent / "training"))
from common import EV, RESULTS, H, Labels, adjust, load_cnn, load_logits, pair_scores, topk, tune  # noqa: E402

STATIC = HERE / "static"
GALLERY_N = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="gui_crop")
    ap.add_argument("--dis", default="gui_dis_attn")
    a = ap.parse_args()

    L = Labels()
    sup = L.host_support()
    crop, dis = (load_logits(RESULTS / "runs" / n / "logits.npz") for n in (a.crop, a.dis))
    cnn = load_cnn()
    dino = {s: pair_scores(crop[s], dis[s], L.cd) for s in EV}
    cnn_pair = {s: pair_scores(cnn[s]["crop"], cnn[s]["disease"], L.cd) for s in EV}

    best = None
    for w in (0.25, 0.5, 1.0):                                   # same grid as evaluate.py
        sc = {s: dino[s] + w * cnn_pair[s] for s in EV}
        h, b, q = tune(sc, L, sup)
        if best is None or h > best[0]:
            best = (h, b, q, w, sc)
    val_h, b, q, w, sc = best
    print(f"tuned on validation: val H {val_h:.2f}  b {b:.3f}  q {q}  w {w}")

    # ---- brief metrics of the deployed model (test scored once, with the val-tuned b, q, w)
    fin = {s: adjust(sc[s], L.seen, b, sup, q) for s in ("test_seen", "test_unseen")}
    m = {}
    for k in (1, 5):
        se, un = topk(fin["test_seen"], L.y["test_seen"], k), topk(fin["test_unseen"], L.y["test_unseen"], k)
        m[f"top{k}"] = {"seen": se, "unseen": un, "H": H(se, un)}
    dis_un = 100 * (dis["test_unseen"].argmax(1) == torch.tensor(L.dfs["test_unseen"].disease_id.values)).float().mean().item()
    print(json.dumps(m))

    def sparse_cm(split):
        cnt = {}
        for t, p in zip(L.y[split].tolist(), fin[split].argmax(1).tolist()):
            cnt[(t, p)] = cnt.get((t, p), 0) + 1
        return [[t, p, c] for (t, p), c in sorted(cnt.items())]

    base = json.loads((HERE / "cnn_assets" / "summary.json").read_text())["Independent/ConvNeXt-Tiny"]
    calib = {"top1": {"seen": 54.25, "unseen": 56.56, "H": 55.38}, "top5": {"seen": 79.4, "unseen": 86.74, "H": 82.91}}  # REPORT.md baseline row
    rows = [
        {"name": "Baseline: independent ConvNeXt-T, raw", "top1": {"seen": base["test_seen"]["pair"]["top1"], "unseen": base["test_unseen"]["pair"]["top1"], "H": base["harmonic"]["top1"]},
         "top5": {"seen": base["test_seen"]["pair"]["top5"], "unseen": base["test_unseen"]["pair"]["top5"], "H": base["harmonic"]["top5"]},
         "disease_unseen": base["test_unseen"]["disease"]["top1"]},
        {"name": "Baseline + calibration", **calib, "disease_unseen": 14.09},
        {"name": "DINO-CPS (this demo)", **m, "disease_unseen": dis_un, "ours": True},
    ]
    STATIC.joinpath("data").mkdir(parents=True, exist_ok=True)
    (STATIC / "data" / "eval.json").write_text(json.dumps({
        "rows": rows, "cm": {s: sparse_cm(s) for s in fin}, "n": {s: len(L.y[s]) for s in fin},
        "tuned": {"val_H": val_h, "b": b, "q": q, "w": w, "runs": [a.crop, a.dis]}}))

    # ---- pair-score params for app.js
    (STATIC / "data" / "params.json").write_text(json.dumps({
        "b": b, "q": q, "w": w, "seen": L.seen.tolist(), "sup": sup.tolist(), "cd": L.cd.tolist()}))

    # ---- gallery: up to GALLERY_N training images per seen pair, 224 px JPEG
    gal_dir = STATIC / "gallery"
    shutil.rmtree(gal_dir, ignore_errors=True); gal_dir.mkdir(parents=True)
    gallery = {}
    for comp, g in L.dfs["train"].groupby("comp_id"):
        files = []
        for i, p in enumerate(g.sample(min(GALLERY_N, len(g)), random_state=0).path):
            f = f"gallery/{comp}_{i}.jpg"
            Image.open(ROOT / "DPD_Small_clean" / p).convert("RGB").resize((224, 224)).save(STATIC / f, quality=85)
            files.append(f)
        gallery[int(comp)] = files

    # ---- meta + examples (reuse the 16 test images picked for the old GUI)
    old = json.loads((HERE / "cnn_assets" / "meta.json").read_text())
    shutil.copytree(HERE / "cnn_assets" / "examples", STATIC / "examples", dirs_exist_ok=True)
    maps = L.maps
    (STATIC / "data" / "meta.json").write_text(json.dumps({
        "crop": old["crop"], "disease": old["disease"], "comp": old["comp"], "comp_to_crop_disease": L.cd.tolist(),
        "unseen_comp_ids": maps["unseen_comp_ids"], "train_comp_ids": sorted(gallery), "examples": old["examples"],
        "gallery": gallery}))

    # ---- CNN fusion models
    (STATIC / "models").mkdir(exist_ok=True)
    for t in ("crop", "disease"):
        shutil.copy(HERE / "cnn_assets" / "models" / f"{t}_ConvNeXt-Tiny.onnx", STATIC / "models" / f"{t}_ConvNeXt-Tiny.onnx")
    print("done:", sum(len(v) for v in gallery.values()), "gallery images")


if __name__ == "__main__":
    main()
