"""Shared paths, labels, pair scoring and the evaluation protocol for v3_disease.

Protocol (same as vlm_v2): rank all 333 crop-disease pairs; every choice is tuned on val + val_unseen;
test_seen / test_unseen are scored once. H = harmonic mean of seen and unseen accuracy.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "DPD_Small_clean_v2"
RESULTS = HERE / os.environ.get("V3_OUT", "results")   # run_all --smoke uses results_smoke
SYNTH = HERE / os.environ.get("V3_SYNTH", "synthetic")
sys.path.insert(0, str(DATA))
from dataset import load_label_maps, read_split  # noqa: E402

SPLITS = ["train", "val", "val_unseen", "test_seen", "test_unseen"]
EV = ["val", "val_unseen", "test_seen", "test_unseen"]
FIELD_SOURCES = {"PWV2", "FP"}     # in-the-wild photos; all other sources are lab-style

# Botanical family per crop. Used by the host-range prior (recommendation 5). Pathology knowledge, not test data.
FAMILY = {
    "apple": "Rosaceae", "apricot": "Rosaceae", "banana": "Musaceae", "basil": "Lamiaceae", "bean": "Fabaceae",
    "pepper_bell": "Solanaceae", "blackgram": "Fabaceae", "blueberry": "Ericaceae", "brassica": "Brassicaceae",
    "broccoli": "Brassicaceae", "cabbage": "Brassicaceae", "carrot": "Apiaceae", "cashew": "Anacardiaceae",
    "cassava": "Euphorbiaceae", "cauliflower": "Brassicaceae", "celery": "Apiaceae", "cherry": "Rosaceae",
    "chilli": "Solanaceae", "citrus": "Rutaceae", "coffee": "Rubiaceae", "coriander": "Apiaceae", "corn": "Poaceae",
    "cotton": "Malvaceae", "cucumber": "Cucurbitaceae", "eggplant": "Solanaceae", "garlic": "Amaryllidaceae",
    "ginger": "Zingiberaceae", "grape": "Vitaceae", "guava": "Myrtaceae", "lettuce": "Asteraceae",
    "mango": "Anacardiaceae", "maple": "Sapindaceae", "olive": "Oleaceae", "paddy": "Poaceae", "palm": "Arecaceae",
    "papaya": "Caricaceae", "peach": "Rosaceae", "peanut": "Fabaceae", "pear": "Rosaceae", "plum": "Rosaceae",
    "potato": "Solanaceae", "pumpkin": "Cucurbitaceae", "raspberry": "Rosaceae", "rose": "Rosaceae",
    "soybean": "Fabaceae", "squash": "Cucurbitaceae", "strawberry": "Rosaceae", "sugarcane": "Poaceae",
    "sunflower": "Asteraceae", "tea": "Theaceae", "tobacco": "Solanaceae", "tomato": "Solanaceae",
    "walnut": "Juglandaceae", "wheat": "Poaceae", "zucchini": "Cucurbitaceae",
}
BROAD_HOST = 3   # a disease seen on >= 3 crops in train is treated as able to infect any crop


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class Labels:
    """Label maps, split frames, pair index and seen-in-train mask."""

    def __init__(self):
        self.maps = load_label_maps(DATA)
        self.cd = torch.tensor([self.maps["comp_to_crop_disease"][c] for c in range(self.maps["num_comps"])])
        self.dfs = {s: read_split(s, DATA) for s in SPLITS}
        self.y = {s: torch.tensor(self.dfs[s].comp_id.values) for s in SPLITS}
        self.seen = torch.zeros(len(self.cd))
        self.seen[self.dfs["train"].comp_id.unique()] = 1.0
        self.n_crop, self.n_dis = self.maps["num_crops"], self.maps["num_diseases"]

    def field(self, s):
        return torch.tensor(self.dfs[s].source.isin(FIELD_SOURCES).values)

    def host_support(self) -> torch.Tensor:
        """Host-range prior (recommendation 5), from train labels + botany only.
        For pairs with no train images: +1 if the disease was seen in train on a crop of the same botanical
        family (a close relative is a known host), -1 if it was seen on fewer than BROAD_HOST crops and on no
        relative (narrow host range, unlikely to jump), else 0. Seen pairs get 0 (the bias b handles them)."""
        tr = self.dfs["train"]
        crops_of = tr.groupby("disease_id").crop.apply(set).to_dict()
        out = torch.zeros(len(self.cd))
        for i, (c, d) in enumerate(self.cd.tolist()):
            if self.seen[i]:
                continue
            hosts = crops_of.get(d, set())
            fam = FAMILY[self.maps["crop"][c]]
            if any(FAMILY[h] == fam for h in hosts):
                out[i] = 1.0
            elif len(hosts) < BROAD_HOST:
                out[i] = -1.0
        return out


def pair_scores(crop_lp: torch.Tensor, dis_lp: torch.Tensor, cd: torch.Tensor) -> torch.Tensor:
    return crop_lp[:, cd[:, 0]] + dis_lp[:, cd[:, 1]]


def topk(s, y, k):
    return 100 * (s.topk(k, 1).indices == y[:, None]).any(1).float().mean().item()


def H(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def adjust(s, seen, b, sup=None, q=0.0):
    """Calibrated stacking: lower seen pairs by b; add q * host support (recommendation 5)."""
    s = s - b * seen
    return s if sup is None else s + q * sup


def tune(sc: dict, L: Labels, sup=None, n=401):
    """Pick bias b (and prior weight q if sup given) by max H(val T1, val_unseen T1). Validation only."""
    sv, su = sc["val"], sc["val_unseen"]
    r = max((x.max() - x.min()).item() for x in (sv, su))
    qs = [0.0] if sup is None else [0.0, 0.5, 1.0, 2.0, 4.0]
    best = (-1.0, 0.0, 0.0)
    for q in qs:
        for b in np.linspace(0, r, n):
            h = H(topk(adjust(sv, L.seen, b, sup, q), L.y["val"], 1),
                  topk(adjust(su, L.seen, b, sup, q), L.y["val_unseen"], 1))
            if h > best[0] + 1e-9:
                best = (h, float(b), q)
    return best   # (val H, b, q)


def report(name: str, sc: dict, L: Labels, b: float, sup=None, q=0.0, extra: dict | None = None) -> dict:
    r = {"row": name, "bias": round(b, 3), "prior q": q}
    for k in (1, 5):
        a = topk(adjust(sc["test_seen"], L.seen, b, sup, q), L.y["test_seen"], k)
        u = topk(adjust(sc["test_unseen"], L.seen, b, sup, q), L.y["test_unseen"], k)
        r |= {f"seen T{k}": a, f"unseen T{k}": u, f"H T{k}": H(a, u)}
    # field vs lab-style pair Top-1 over both test splits
    ok = torch.cat([adjust(sc[s], L.seen, b, sup, q).argmax(1) == L.y[s] for s in ("test_seen", "test_unseen")])
    fld = torch.cat([L.field(s) for s in ("test_seen", "test_unseen")])
    r["field T1"] = 100 * ok[fld].float().mean().item()
    r["lab T1"] = 100 * ok[~fld].float().mean().item()
    return r | (extra or {})


def load_cnn():
    """Team ConvNeXt-Tiny independent crop/disease logits (exported from vlm_v2 cache). None if missing."""
    f = HERE / "assets" / "cnn_logits.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {s: {t: torch.log_softmax(torch.from_numpy(z[f"{s}_{t}"]).float(), 1) for t in ("crop", "disease")} for s in EV}


def save_logits(path: Path, out: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v.float().numpy().astype(np.float16) for k, v in out.items()})


def load_logits(path: Path) -> dict:
    z = np.load(path)
    return {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


if __name__ == "__main__":   # self-check: prior + scoring shapes
    L = Labels()
    sup = L.host_support()
    print("pairs", len(L.cd), "seen", int(L.seen.sum()), "unseen", int((1 - L.seen).sum()))
    assert sup[L.seen.bool()].abs().sum() == 0
    s = pair_scores(torch.zeros(3, L.n_crop), torch.zeros(3, L.n_dis), L.cd)
    assert s.shape == (3, len(L.cd))
    print("ok")
