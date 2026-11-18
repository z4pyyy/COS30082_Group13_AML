"""
export_onnx.py - Export the 9 baseline checkpoints for the in-browser validation GUI (web/).

    python3.14 export_onnx.py              # export + evaluate all, writes web/
    python3.14 export_onnx.py --skip-export  # re-run evaluation only

Every ONNX model has input "input" [B,3,224,224] and, per head it has, two outputs:
    <head>_probs [B, C]        softmax probabilities         (head = crop | disease)
    <head>_cam   [B, C, h, w]  class activation map per class
Independent crop model -> crop_*. Independent disease model -> disease_*. Multitask -> both.
The page merges the outputs of 1 (multitask) or 2 (independent) models the same way,
so pair scoring is identical for both approaches:
    pair_score[comp] = log p(crop[comp]) + log p(disease[comp])     (333 comps)

Evaluation runs the exported ONNX files (what the page runs), not PyTorch, on
val / val_unseen / test_seen / test_unseen, and stores metrics + sparse confusion matrices.
A parity check compares ONNX with PyTorch on a batch of test images first.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import timm
import torch
import torch.nn as nn
from onnx import helper, numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "DPD_Small_clean_v2"
WEB = ROOT / "web"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 224
BACKBONES = ["ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny"]
SPLITS = {"test_seen": "test_seen.csv", "test_unseen": "test_unseen.csv", "val": "val.csv", "val_unseen": "val_unseen.csv"}

# file stem -> checkpoint path
CKPTS = {}
for b in BACKBONES:
    CKPTS[f"crop_{b}"] = ROOT / "crop_independent_v2/output/models" / f"{b}_best.pt"
    CKPTS[f"disease_{b}"] = ROOT / "disease_inde_cnn_v2/output/models" / f"{b}_best.pt"
    CKPTS[f"mt_{b}"] = ROOT / "multitask_cnn/output/models" / f"{b}_best.pt"

# GUI option -> ONNX files it needs
OPTIONS = {}
for b in BACKBONES:
    OPTIONS[f"Independent/{b}"] = [f"crop_{b}", f"disease_{b}"]
    OPTIONS[f"Multitask/{b}"] = [f"mt_{b}"]


class Wrapper(nn.Module):
    """backbone (timm, num_classes=0) + one Linear per head -> probs and CAM per head."""

    def __init__(self, backbone, heads: dict):
        super().__init__()
        self.backbone = backbone.eval()
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        f = self.backbone.forward_features(x)                      # [B, D, h, w]
        b, d, h, w = f.shape
        pooled = self.backbone.forward_head(f)                     # [B, D] (pool, + LayerNorm for ConvNeXt)
        # CAM: run the same pooling/norm + head on every spatial position as a 1x1 map.
        per_pos = self.backbone.forward_head(f.permute(0, 2, 3, 1).reshape(b * h * w, d, 1, 1))
        out = []
        for head in self.heads.values():
            out.append(head(pooled).softmax(1))
            out.append(head(per_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2))
        return tuple(out)


def load_wrapper(stem):
    c = torch.load(CKPTS[stem], map_location="cpu", weights_only=False)
    if stem.startswith("mt_"):
        bb = timm.create_model(c["args"]["backbone"][5:], pretrained=False, num_classes=0)
        sd = c["model"]
        n_crop, n_dis = sd["crop_head.weight"].shape[0], sd["disease_head.weight"].shape[0]
        heads = {"crop": nn.Linear(bb.num_features, n_crop), "disease": nn.Linear(bb.num_features, n_dis)}
        bb.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")})
        for h in heads:
            heads[h].load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith(f"{h}_head.")})
        return Wrapper(bb, heads).eval(), c.get("epoch")
    full = timm.create_model(c["timm_id"], pretrained=False, num_classes=int(c["num_classes"]))
    full.load_state_dict(c["state_dict"])
    head = full.get_classifier()
    full.reset_classifier(0)                                       # keeps pooling/norm, drops the Linear
    return Wrapper(full, {c["task"]: head}).eval(), c.get("epoch")


def fp16_weights(path):
    """Store big float32 weights as float16 + Cast back to float32 (half the file, same compute)."""
    m = onnx.load(path)
    g = m.graph
    inits, casts = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 1024:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
        else:
            inits.append(init)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def load_image(path):
    im = Image.open(path).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BILINEAR)              # plain resize, same as dataset build
    a = np.asarray(im, dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def export(stem, out_dir):
    model, epoch = load_wrapper(stem)
    names = [f"{h}_{k}" for h in model.heads for k in ("probs", "cam")]
    path = out_dir / f"{stem}.onnx"
    torch.onnx.export(model, (torch.randn(1, 3, SIZE, SIZE),), str(path), input_names=["input"],
                      output_names=names, opset_version=17, dynamo=False,
                      dynamic_axes={"input": {0: "b"}, **{n: {0: "b"} for n in names}})
    fp16_weights(str(path))
    # parity: ONNX (fp16 weights) vs PyTorch on 32 test_seen images
    df = pd.read_csv(DATA / "test_seen.csv").head(32)
    x = np.stack([load_image(DATA / p) for p in df.path])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    o = sess.run(None, {"input": x})
    with torch.no_grad():
        t = model(torch.from_numpy(x))
    probs_o, probs_t = o[0], t[0].numpy()
    par = {"top1_agree": float((probs_o.argmax(1) == probs_t.argmax(1)).mean()),
           "max_abs_prob_diff": float(np.abs(probs_o - probs_t).max())}
    mb = round(path.stat().st_size / 1e6, 1)
    print(f"  {stem}: {mb} MB, cam {o[1].shape[2:]}, parity {par}")
    return {"file": f"models/{stem}.onnx", "mb": mb, "heads": list(model.heads), "epoch": epoch, "parity": par}


def run_onnx(path, xs, bs=64):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.get_outputs()
    keep = [i for i, o in enumerate(outs) if o.name.endswith("_probs")]
    res = {outs[i].name[:-6]: [] for i in keep}
    for s in range(0, len(xs), bs):
        o = sess.run([outs[i].name for i in keep], {"input": xs[s:s + bs]})
        for i, arr in zip(keep, o):
            res[outs[i].name[:-6]].append(arr)
    return {k: np.concatenate(v) for k, v in res.items()}


def metrics(scores, y, ks=(1, 3, 5)):
    top = np.argsort(-scores, 1)[:, :max(ks)]
    hit = top == y[:, None]
    m = {f"top{k}": round(float(hit[:, :k].any(1).mean() * 100), 2) for k in ks}
    pred = top[:, 0]
    m["per_class_top1"] = round(float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]) * 100), 2)
    return m, pred


def sparse_cm(y, pred):
    pairs, counts = np.unique(np.stack([y, pred], 1), axis=0, return_counts=True)
    return [[int(a), int(b), int(c)] for (a, b), c in zip(pairs, counts)]


def evaluate(maps, cd):
    cache = {}
    xs = {}
    for split, csv in SPLITS.items():
        df = pd.read_csv(DATA / csv)
        xs[split] = (df, np.stack([load_image(DATA / p) for p in df.path]))
        print(f"  loaded {split}: {len(df)} images")
    for stem in CKPTS:
        for split, (df, x) in xs.items():
            cache[stem, split] = run_onnx(WEB / "models" / f"{stem}.onnx", x)
        print(f"  ran {stem}")

    results = {}
    for opt, stems in OPTIONS.items():
        r = {}
        for split, (df, _) in xs.items():
            probs = {}
            for s in stems:
                probs.update(cache[s, split])
            lc, ld = np.log(probs["crop"] + 1e-9), np.log(probs["disease"] + 1e-9)
            pair = lc[:, cd[:, 0]] + ld[:, cd[:, 1]]
            y = {"pair": df.comp_id.values, "crop": df.crop_id.values, "disease": df.disease_id.values}
            sc = {"pair": pair, "crop": lc, "disease": ld}
            r[split] = {"n": len(df), "metrics": {}, "cm": {}}
            for h in ("pair", "crop", "disease"):
                m, pred = metrics(sc[h], y[h])
                r[split]["metrics"][h] = m
                r[split]["cm"][h] = sparse_cm(y[h], pred)
        hm = lambda a, b: round(2 * a * b / (a + b), 2) if a + b else 0.0
        r["harmonic"] = {k: hm(r["test_seen"]["metrics"]["pair"][k], r["test_unseen"]["metrics"]["pair"][k])
                         for k in ("top1", "top3", "top5")}
        results[opt] = r
        print(f"  {opt}: pair seen {r['test_seen']['metrics']['pair']['top1']} "
              f"unseen {r['test_unseen']['metrics']['pair']['top1']} H {r['harmonic']['top1']}")
    return results


def pick_examples(n=8, seed=7):
    out = []
    rng = np.random.default_rng(seed)
    ex = WEB / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    for split in ("test_seen", "test_unseen"):
        df = pd.read_csv(DATA / SPLITS[split])
        for i in rng.choice(len(df), size=n, replace=False):
            row = df.iloc[i]
            name = Path(row.path).name
            shutil.copy(DATA / row.path, ex / name)
            out.append({"file": f"examples/{name}", "split": split, "comp": int(row.comp_id)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-export", action="store_true")
    a = p.parse_args()
    maps = json.loads((DATA / "label_maps.json").read_text())
    cd = np.array([maps["comp_to_crop_disease"][str(c)] for c in range(maps["num_comps"])])
    (WEB / "models").mkdir(parents=True, exist_ok=True)

    info_path = WEB / "models.json"
    if a.skip_export:
        files = json.loads(info_path.read_text())["files"]
    else:
        print("export:")
        files = {s: export(s, WEB / "models") for s in CKPTS}
    print("evaluate (ONNX):")
    results = evaluate(maps, cd)

    meta = {
        "img_size": SIZE, "mean": MEAN, "std": STD,
        "crop": [maps["crop"][str(i)] for i in range(maps["num_crops"])],
        "disease": [maps["disease"][str(i)] for i in range(maps["num_diseases"])],
        "comp": [maps["comp"][str(i)] for i in range(maps["num_comps"])],
        "comp_to_crop_disease": cd.tolist(),
        "unseen_comp_ids": maps["unseen_comp_ids"],
        "examples": pick_examples(),
        "options": {k: v for k, v in OPTIONS.items()},
    }
    (WEB / "meta.json").write_text(json.dumps(meta))
    info_path.write_text(json.dumps({"files": files}, indent=1))
    (WEB / "results").mkdir(exist_ok=True)
    summary = {}
    for opt, r in results.items():
        (WEB / "results" / f"{opt.replace('/', '_')}.json").write_text(json.dumps(r, separators=(",", ":")))
        summary[opt] = {"harmonic": r["harmonic"], **{s: r[s]["metrics"] for s in SPLITS}}
    (WEB / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote web/meta.json, web/models.json, web/summary.json, web/results/, web/examples/")


if __name__ == "__main__":
    main()
