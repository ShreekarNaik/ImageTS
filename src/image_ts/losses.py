"""Losses for Image-TS training."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from image_ts import TriangleBatch
from image_ts.config import LossConfig
from image_ts.geometry import triangle_area

Tensor = torch.Tensor


def weighted_l1(pred: Tensor, target: Tensor, importance: Optional[Tensor]) -> Tensor:
    diff = torch.abs(pred - target)
    if importance is not None:
        diff = diff * importance[..., None]
    return diff.mean()


def _ssim(x: Tensor, y: Tensor, window: int = 11) -> Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    padding = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return torch.clamp(numerator / denominator, 0.0, 1.0)


def ms_ssim(pred: Tensor, target: Tensor, levels: int = 3) -> Tensor:
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    weights = torch.tensor([0.0448, 0.2856, 0.3001], device=pred.device)
    mssim = []
    for _ in range(levels):
        score = _ssim(x, y)
        mssim.append(score.mean())
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)
    return torch.sum(weights[: len(mssim)] * torch.stack(mssim))


def area_regularizer(triangles: TriangleBatch) -> Tensor:
    return triangle_area(triangles.vertices).mean()


def sigma_regularizer(triangles: TriangleBatch) -> Tensor:
    return torch.mean(torch.square(torch.log(triangles.sigma + 1e-4)))


def reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    triangles: TriangleBatch,
    config: LossConfig,
    importance: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    l1 = weighted_l1(pred, target, importance)
    ms = 1.0 - ms_ssim(pred, target)
    area_reg = area_regularizer(triangles)
    sigma_reg = sigma_regularizer(triangles)
    total = (
        config.lambda_l1 * l1
        + config.lambda_ms_ssim * ms
        + config.lambda_area * area_reg
        + config.lambda_sigma * sigma_reg
    )
    return {
        "total": total,
        "l1": l1,
        "ms_ssim": ms,
        "area": area_reg,
        "sigma": sigma_reg,
    }
