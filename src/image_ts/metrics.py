"""Evaluation metrics."""
from __future__ import annotations

import math
from typing import Dict

import torch

from image_ts.losses import _ssim as _loss_ssim


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    return _loss_ssim(x, y).mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    if pred.device != target.device:
        target = target.to(pred.device)
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
