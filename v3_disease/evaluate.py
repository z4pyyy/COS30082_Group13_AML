"""Combine the saved crop / disease log-probs into 333-pair scores and report everything.

Per seed and per disease model:
  calib         pair = log p(crop) + log p(disease), seen pairs lowered by bias b
  +prior        + q * host-range support (recommendation 5)
  +CNN          + w * team ConvNeXt-T independent pair score
  +prior+CNN    both
DINOv2 disease models are also fused with the VL symptom-text model ("+VL").
Every b, q, w, fusion weight is tuned on val + val_unseen. Final model = highest mean val H over seeds.

Writes results/summary_all.csv, results/summary.csv, results/heads.csv, results/REPORT.md, results/final_cm/.
Run: python v3_disease/evaluate.py
"""
import json
import re

import pandas as pd
import torch

from common import RESULTS, Labels, load_cnn, load_logits, pair_scores, report, tune, EV, adjust
from metrics import evaluate_split

NAMES = {
    "dis_base": "D0 DINOv2 fine-tuned, CLS head",
    "dis_attn": "D1 + attention lesion pooling (R1)",
    "dis_attn_adv": "D2 + crop adversary (R1+R2)",
    "dis_attn_adv_syn": "D3 + synthetic pairs (R1+R2+R4)",
    "dis_full": "D4 + field robustness (R1+R2+R4+R6)",
    "vl": "VL symptom-text, fine-tuned (R3)",
    "vl_zeroshot": "VL symptom-text, zero-shot (R3)",
}


def runs():
    out = {}
    for d in (RESULTS / "runs").glob("*"):
        if (d / "logits.npz").exists():
            out[d.name] = load_logits(d / "logits.npz")
    return out


def variants(name, dis, crop, cnn_pair, sup, L):
    sc = {s: pair_scores(crop[s], dis[s], L.cd) for s in EV}
    rows = []
    for v, use_sup, use_cnn in [("calib", False, False), ("+prior", True, False),
                                ("+CNN", False, True), ("+prior+CNN", True, True)]:
        if use_cnn and cnn_pair is None:
            continue
        best = None
        for w in ([0.25, 0.5, 1.0] if use_cnn else [0.0]):
            s2 = {s: sc[s] + w * cnn_pair[s] for s in EV} if use_cnn else sc
            h, b, q = tune(s2, L, sup if use_sup else None)
            if best is None or h > best[0]:
                best = (h, b, q, w, s2)
        h, b, q, w, s2 = best
        rows.append(report(f"{name} | {v}", s2, L, b, sup if use_sup else None, q,
                           {"model": name, "variant": v, "val H": h, "cnn w": w, "_scores": s2}))
    return rows


def main():
    L = Labels()
    R = runs()
    sup = L.host_support()
    cnn = load_cnn()
    cnn_pair = None if cnn is None else {s: pair_scores(cnn[s]["crop"], cnn[s]["disease"], L.cd) for s in EV}
    dis_acc = lambda lp: 100 * (lp["test_unseen"].argmax(1) == torch.tensor(L.dfs["test_unseen"].disease_id.values)).float().mean().item()

    rows = []
    if cnn_pair is not None:   # reference: team CNN Product-calib (v2 baseline)
        h, b, _ = tune(cnn_pair, L)
        rows.append(report("baseline | Product-calib (CNN)", cnn_pair, L, b, extra={
            "model": "baseline", "variant": "Product-calib (CNN)", "val H": h, "seed": -1,
            "disease acc test_unseen": dis_acc({s: cnn[s]["disease"] for s in EV})}))
    seeds = sorted({int(m.group(1)) for k in R if (m := re.fullmatch(r"crop_s(\d+)", k))})
    for sd in seeds:
        crop = R[f"crop_s{sd}"]
        models = {k: R[f"{k}_s{sd}"] for k in NAMES if f"{k}_s{sd}" in R}
        if "vl_zeroshot" in R:
            models["vl_zeroshot"] = R["vl_zeroshot"]
        vl = models.get("vl")
        for k in [k for k in list(models) if k.startswith("dis_")] if vl is not None else []:
            best = None
            for a in (0.25, 0.5, 1.0):   # disease fusion weight with VL, tuned on val disease H
                f = {s: torch.log_softmax(models[k][s] + a * vl[s], 1) for s in EV}
                acc = [100 * (f[s].argmax(1) == torch.tensor(L.dfs[s].disease_id.values)).float().mean().item()
                       for s in ("val", "val_unseen")]
                hh = 2 * acc[0] * acc[1] / max(acc[0] + acc[1], 1e-9)
                if best is None or hh > best[0]:
                    best = (hh, f)
            models[f"{k}+vl"] = best[1]
        for k, dis in models.items():
            for r in variants(k, dis, crop, cnn_pair, sup, L):
                rows.append(r | {"seed": sd, "disease acc test_unseen": dis_acc(dis)})

    scores = {(r["model"], r["variant"], r["seed"]): r.pop("_scores") for r in rows if "_scores" in r}
    df = pd.DataFrame(rows)
    df.round(2).to_csv(RESULTS / "summary_all.csv", index=False)
    cols = ["val H", "seen T1", "unseen T1", "H T1", "seen T5", "unseen T5", "H T5", "field T1", "lab T1",
            "disease acc test_unseen"]
    g = df.groupby(["model", "variant"])[cols]
    summ = g.mean().round(2).join(g.std().round(2), rsuffix=" sd").join(g.size().rename("seeds"))
    summ = summ.sort_values("val H", ascending=False)
    summ.to_csv(RESULTS / "summary.csv")

    heads = []
    for d in sorted((RESULTS / "runs").glob("*/meta.json")):
        m = json.loads(d.read_text())
        heads.append({"run": d.parent.name, "task": m.get("task"), "best epoch": m.get("best epoch"),
                      **{k: round(v, 2) for k, v in m.items() if k.endswith(" acc") and isinstance(v, float)}})
    pd.DataFrame(heads).to_csv(RESULTS / "heads.csv", index=False)

    # final model: best mean val H (baseline excluded); confusion matrices from its first seed
    pick = summ.drop(index="baseline", level=0, errors="ignore").index[0]
    sd = df[(df.model == pick[0]) & (df.variant == pick[1])].seed.min()
    r = df[(df.model == pick[0]) & (df.variant == pick[1]) & (df.seed == sd)].iloc[0]
    s2 = scores[(pick[0], pick[1], sd)]
    q = r["prior q"]
    names = [L.maps["comp"][i] for i in range(len(L.cd))]
    cmdir = RESULTS / "final_cm"
    for s, tag in (("test_seen", "seen"), ("test_unseen", "unseen")):
        adj = adjust(s2[s], L.seen, r["bias"], sup if q else None, q)
        evaluate_split(adj, L.y[s], names, str(cmdir), prefix=f"{pick[0]}_{pick[1]}".replace("+", "plus").replace(" ", ""),
                       split=tag, paths=L.dfs[s].path.tolist())

    with open(RESULTS / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("# v3_disease results\n\nMean over seeds (sd = standard deviation). Sorted by val H "
                "(the selection score). Test is never used for a choice.\n\n")
        f.write(f"**Final model (best mean val H): `{pick[0]}` / `{pick[1]}`**: {NAMES.get(pick[0].split('+')[0], pick[0])}\n\n")
        f.write(summ.reset_index().to_markdown(index=False))
        f.write("\n\n## Per-run head accuracy (task accuracy, %)\n\n")
        f.write(pd.DataFrame(heads).to_markdown(index=False))
        f.write("\n\nKey honest metric: `disease acc test_unseen` = disease head alone on unseen-pair images "
                "(vlm_v2 probe: 4.6%). If it rises, the model learned the disease, not the crop shortcut.\n")
    print(summ.head(25).to_string())
    print(f"\nfinal: {pick}  -> {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
