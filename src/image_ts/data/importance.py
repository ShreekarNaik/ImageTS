"""Importance map utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_importance(path: Optional[Path], shape: tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.ones(shape, dtype=torch.float32)
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype("float32")
    tensor = torch.from_numpy(arr)
    tensor = tensor / (tensor.max() + 1e-8)
    return F.interpolate(tensor[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def gradient_importance(image: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # image: (H, W, C)
    tensor = image.permute(2, 0, 1).unsqueeze(0)
    dx = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    dy = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    dx = F.pad(dx, (0, 0, 0, 1))
    dy = F.pad(dy, (0, 1, 0, 0))
    mag = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)
    importance = mag.mean(dim=1, keepdim=False)[0]
    importance = importance / (importance.max() + eps)
    return importance
