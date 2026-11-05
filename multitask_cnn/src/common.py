"""Shared paths, seeding and label-map helpers."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dataset" / "DPD_Small_clean_v2"
RUNS_DIR = PROJECT_ROOT / "runs"

# Make `from DPD_Small_clean_v2.dataset import ...` work from anywhere.
if str(DATA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT.parent))

from DPD_Small_clean_v2.dataset import get_loaders, load_label_maps  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def comp_index(maps: dict) -> torch.Tensor:
    """(num_comps, 2) tensor: comp id -> [crop id, disease id]."""
    return torch.tensor([maps["comp_to_crop_disease"][c] for c in range(maps["num_comps"])], dtype=torch.long)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


__all__ = ["PROJECT_ROOT", "DATA_ROOT", "RUNS_DIR", "get_loaders", "load_label_maps",
           "seed_everything", "comp_index", "save_json"]
