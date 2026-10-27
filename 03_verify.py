"""Step 8: verify DPD_Small_clean/ before training.

Checks: files exist, no split overlap, label maps consistent, loaders work, GPU forward pass.
Writes data_prep/reports/augmentation_preview.png (column 1 = no augmentation).
Run from the project root: python data_prep/03_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DPD_Small_clean.dataset import (IMAGENET_MEAN, IMAGENET_STD, SPLITS, DPDDataset,  # noqa: E402
                                     build_transforms, class_weights, get_loaders, load_label_maps)

CLEAN = ROOT / "DPD_Small_clean"
REPORTS = ROOT / "data_prep" / "reports"


def main() -> None:
    maps = load_label_maps()
    print(f"label maps: {maps['num_crops']} crops, {maps['num_diseases']} diseases, "
          f"{maps['num_comps']} comps ({len(maps['unseen_comp_ids'])} unseen)")

    dfs = {s: pd.read_csv(CLEAN / f"{s}.csv") for s in SPLITS}
    for s, df in dfs.items():
        missing = (~df.path.map(lambda p: (CLEAN / p).exists())).sum()
        pairs = df[["comp_id", "crop_id", "disease_id"]].drop_duplicates()
        bad = sum(maps["comp_to_crop_disease"][c] != [a, b] for c, a, b in pairs.values)
        assert missing == 0 and bad == 0, f"{s}: {missing} missing files, {bad} bad label rows"
    names = {s: set(df.filename) for s, df in dfs.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not names[a] & names[b], f"overlap {a} & {b}"
    assert not set(dfs["train"].comp_id) & set(maps["unseen_comp_ids"]), "unseen comp in train"
    print("files, labels, split overlap: OK")

    loaders = get_loaders(batch_size=32, num_workers=4, balance_by="comp_id")
    for s, dl in loaders.items():
        x, y = next(iter(dl))
        print(f"  {s:12s} n={len(dl.dataset):5d} batch={tuple(x.shape)} crop/disease/comp max="
              f"{y['crop'].max().item()}/{y['disease'].max().item()}/{y['comp'].max().item()}")

    w = class_weights(dfs["train"], "disease_id", maps["num_diseases"])
    print(f"disease class weights (power 0.5): min {w.min():.3f} max {w.max():.3f}")

    aug, plain = DPDDataset("train"), DPDDataset("train", transform=build_transforms(False))
    idx = aug.df.index[aug.df.comp == "tomato_late_blight"][0]
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    grid = [plain[idx][0]] + [aug[idx][0] for _ in range(7)]
    save_image(torch.stack([g * std + mean for g in grid]).clamp(0, 1),
               REPORTS / "augmentation_preview.png", nrow=8, padding=4)
    print("augmentation preview: data_prep/reports/augmentation_preview.png")

    if torch.cuda.is_available():
        for name in ["resnet50", "efficientnet_b3", "convnext_tiny"]:
            net = getattr(torchvision.models, name)(weights=None).cuda().eval()
            with torch.no_grad(), torch.autocast("cuda"):
                out = net(x.cuda())
            print(f"  GPU forward {name}: {tuple(out.shape)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
