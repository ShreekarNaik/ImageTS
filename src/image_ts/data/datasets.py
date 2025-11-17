"""Image loading utilities for Image-TS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image
import torch


@dataclass
class LoadedImage:
    path: Path
    tensor: torch.Tensor  # (H, W, C)


class ImageFolderDataset:
    def __init__(self, paths: Iterable[Path]):
        self.paths: List[Path] = sorted(Path(p) for p in paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> LoadedImage:
        path = self.paths[idx]
        tensor = load_image(path)
        return LoadedImage(path=path, tensor=tensor)


def load_image(path: Path, *, channels: int = 3) -> torch.Tensor:
    image = Image.open(path).convert("RGB" if channels == 3 else "RGBA")
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr)
    return tensor


def save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = (tensor.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def list_images(root: Path, patterns: Sequence[str] = ("*.png", "*.jpg", "*.jpeg")) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(files)
