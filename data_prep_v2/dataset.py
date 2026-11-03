"""PyTorch loader for DPD_Small_clean_v2.

    import sys; sys.path.insert(0, "<folder that contains DPD_Small_clean_v2>")
    from DPD_Small_clean_v2.dataset import get_loaders, load_label_maps
    loaders = get_loaders(batch_size=64, num_workers=2)
    for x, y in loaders["train"]:      # x: (B,3,224,224) ImageNet-normalised; y: {"crop","disease","comp"} (LongTensor)
        ...

Splits: train, val, val_unseen, test_seen, test_unseen.
Augmentation is applied on the fly to the train split only. The images on disk are NOT augmented.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "val_unseen", "test_seen", "test_unseen")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE = 224
LABELS = ("crop", "disease", "comp")


def build_transforms(train: bool) -> T.Compose:
    """Train: light geometric + colour augmentation. Eval: tensor + normalise only (images are already 224x224)."""
    if not train:
        return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([
        T.RandomResizedCrop(SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def load_label_maps(root: Path | str = ROOT) -> dict:
    """label_maps.json with integer keys restored (JSON stores them as strings)."""
    maps = json.loads((Path(root) / "label_maps.json").read_text(encoding="utf-8"))
    for k in ("crop", "disease", "comp", "comp_to_crop_disease"):
        maps[k] = {int(i): v for i, v in maps[k].items()}
    return maps


def read_split(split: str, root: Path | str = ROOT) -> pd.DataFrame:
    assert split in SPLITS, f"unknown split {split!r}, expected one of {SPLITS}"
    return pd.read_csv(Path(root) / f"{split}.csv")


class DPDDataset(Dataset):
    """One split. Returns (image tensor, {"crop": id, "disease": id, "comp": id})."""

    def __init__(self, split: str, transform=None, root: Path | str = ROOT):
        self.root, self.split = Path(root), split
        self.df = read_split(split, root)
        self.transform = transform if transform is not None else build_transforms(split == "train")
        self._y = {k: torch.tensor(self.df[f"{k}_id"].values, dtype=torch.long) for k in LABELS}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        with Image.open(self.root / self.df.path.iat[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, {k: self._y[k][i] for k in LABELS}


def class_weights(df: pd.DataFrame | None = None, col: str = "disease_id", num_classes: int | None = None,
                  power: float = 0.5, root: Path | str = ROOT) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss(weight=...). Same call as v1:
    class_weights(train_df, "disease_id", num_diseases). df defaults to the train split, num_classes to the label map.
    Classes absent from df get 0. power=0.5 is a softened (sqrt) re-weighting. Mean weight over df's images is 1."""
    df = read_split("train", root) if df is None else df
    if num_classes is None:
        maps = load_label_maps(root)
        num_classes = {"crop_id": maps["num_crops"], "disease_id": maps["num_diseases"], "comp_id": maps["num_comps"]}[col]
    y = torch.tensor(df[col].values)
    counts = torch.bincount(y, minlength=num_classes).float()
    w = torch.where(counts > 0, counts.clamp(min=1) ** -power, torch.zeros_like(counts))
    return w / w[y].mean()


def balanced_sampler(df: pd.DataFrame, by: str = "comp_id", power: float = 0.5) -> WeightedRandomSampler:
    """Over-sample rare classes. power=1 is fully balanced, 0.5 is a softer version."""
    freq = df[by].map(df[by].value_counts())
    w = torch.tensor((1.0 / freq.values) ** power, dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(df), replacement=True)


def get_loaders(batch_size: int = 64, num_workers: int = 2, balance_by: str | None = None,
                root: Path | str = ROOT, splits=SPLITS) -> dict[str, DataLoader]:
    """DataLoaders for the requested splits. Only train is shuffled and augmented.
    balance_by="comp_id" | "disease_id" | "crop_id" switches train to a balanced sampler.
    Use either balance_by or class_weights() in the loss, not both."""
    pin = torch.cuda.is_available()
    out = {}
    for s in splits:
        ds = DPDDataset(s, root=root)
        train = s == "train"
        sampler = balanced_sampler(ds.df, balance_by) if (train and balance_by) else None
        out[s] = DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin, drop_last=train,
                            persistent_workers=num_workers > 0)
    return out


def comp_index(root: Path | str = ROOT) -> torch.Tensor:
    """(num_comps, 2): comp id -> [crop id, disease id]. Used to score crop+disease pairs:
    score = log_softmax(crop)[:, cd[:, 0]] + log_softmax(disease)[:, cd[:, 1]]."""
    maps = load_label_maps(root)
    return torch.tensor([maps["comp_to_crop_disease"][c] for c in range(maps["num_comps"])], dtype=torch.long)
