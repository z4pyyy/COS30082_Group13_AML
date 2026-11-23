"""Recommendation 4: synthetic new crop-disease combinations by lesion cut-paste (train images only).

For each sample: take a diseased train image (disease d on crop A), cut its lesion pixels, paste them onto a
healthy train leaf of another crop B where d was never seen on B in train, but B is a plausible host
(same botanical family as a known host, or d is seen on >= BROAD_HOST crops). Label: disease d, crop B.
The disease model then sees the same lesions on different leaves, which breaks the "crop -> usual disease" shortcut.

Lesion mask (no annotations exist): pixels inside the leaf area whose hue is not green, or that are dark.
Crude by design; the preview image shows what it produces.

Run: python v3_disease/make_synthetic.py [--n 4000]
"""
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from common import BROAD_HOST, DATA, FAMILY, FIELD_SOURCES, RESULTS, SYNTH, Labels

GREEN = (38, 100)   # PIL HSV hue range (0-255) treated as green leaf tissue


def masks(im: Image.Image):
    hsv = np.asarray(im.convert("HSV")).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= GREEN[0]) & (h <= GREEN[1]) & (s > 50) & (v > 40)
    # leaf area = green tissue grown by 15 px, so lesions next to green count
    area = np.asarray(Image.fromarray((green * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(15))) > 0
    lesion = area & ~green & (((s > 60) & (v > 30)) | (v < 70))
    return green, lesion


def paste(src: Image.Image, dst: Image.Image, rng: random.Random):
    _, les = masks(src)
    g_dst, _ = masks(dst)
    if not (0.02 < les.mean() < 0.45) or g_dst.mean() < 0.15:
        return None
    dx, dy = rng.randint(-40, 40), rng.randint(-40, 40)
    les = np.roll(np.roll(les, dy, 0), dx, 1)
    src_np = np.roll(np.roll(np.asarray(src), dy, 0), dx, 1)
    keep = les & (np.asarray(Image.fromarray((g_dst * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0)
    if keep.mean() < 0.01:
        return None
    a = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))) / 255.0
    out = np.asarray(dst) * (1 - a[..., None]) + src_np * a[..., None]
    return Image.fromarray(out.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    L = Labels()
    tr = L.dfs["train"]
    hosts = tr.groupby("disease_id").crop.apply(set).to_dict()
    healthy = tr[tr.disease == "healthy"].groupby("crop_id").path.apply(list).to_dict()
    seen_pairs = set(zip(tr.crop_id, tr.disease_id))
    # lesion sources: lab-style photos only (field photos give shadows and soil as "lesions")
    diseased = tr[(tr.disease != "healthy") & ~tr.source.isin(FIELD_SOURCES)].reset_index(drop=True)

    def targets(d):
        fams = {FAMILY[h] for h in hosts[d]}
        broad = len(hosts[d]) >= BROAD_HOST
        return [c for c in healthy if (c, d) not in seen_pairs and (broad or FAMILY[L.maps["crop"][c]] in fams)]

    tgt = {d: targets(d) for d in diseased.disease_id.unique()}
    pool = diseased[diseased.disease_id.map(lambda d: len(tgt[d]) > 0)].reset_index(drop=True)
    print(f"diseases with a plausible new host: {pool.disease_id.nunique()} / {diseased.disease_id.nunique()}")
    SYNTH.mkdir(parents=True, exist_ok=True)
    rows, tries = [], 0
    while len(rows) < args.n and tries < args.n * 20:
        tries += 1
        r = pool.iloc[rng.randrange(len(pool))]
        c = rng.choice(tgt[r.disease_id])
        src = Image.open(DATA / r.path).convert("RGB")
        dst = Image.open(DATA / rng.choice(healthy[c])).convert("RGB")
        out = paste(src, dst, rng)
        if out is None:
            continue
        name = f"syn_{len(rows):05d}.jpg"
        out.save(SYNTH / name, quality=92)
        rows.append({"path": f"{SYNTH.name}/{name}", "crop_id": c, "disease_id": int(r.disease_id), "comp_id": -1,
                     "source": "SYN", "src_crop_id": int(r.crop_id)})
    df = pd.DataFrame(rows)
    df.to_csv(SYNTH / "index.csv", index=False)
    print(f"wrote {len(df)} images ({tries} tries), {df.disease_id.nunique()} diseases, {df.crop_id.nunique()} crops")

    # preview for the report: 8 random samples
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.5))
    for a, (_, r) in zip(ax.flat, df.sample(min(8, len(df)), random_state=0).iterrows()):
        a.imshow(Image.open(SYNTH.parent / r.path))
        a.set_title(f"{L.maps['disease'][r.disease_id]}\non {L.maps['crop'][r.crop_id]} "
                    f"(from {L.maps['crop'][r.src_crop_id]})", fontsize=8)
        a.axis("off")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "synthetic_preview.png", dpi=110)


if __name__ == "__main__":
    main()
