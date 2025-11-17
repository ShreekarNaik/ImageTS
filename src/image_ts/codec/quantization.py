"""Quantization utilities for Image-TS triangles."""
from __future__ import annotations

from typing import Tuple

import torch

SIGMA_RANGE: Tuple[float, float] = (1e-4, 0.2)


def _scale(bits: int) -> int:
    return (1 << bits) - 1


def quantize_positions(vertices: torch.Tensor, bits: int) -> torch.Tensor:
    scale = _scale(bits)
    return torch.round(vertices.clamp(0.0, 1.0) * scale).to(torch.int32)


def dequantize_positions(values: torch.Tensor, bits: int) -> torch.Tensor:
    scale = _scale(bits)
    return values.float() / scale


def quantize_colors(colors: torch.Tensor, bits: int) -> torch.Tensor:
    scale = _scale(bits)
    return torch.round(colors.clamp(0.0, 1.0) * scale).to(torch.int32)


def dequantize_colors(values: torch.Tensor, bits: int) -> torch.Tensor:
    scale = _scale(bits)
    return values.float() / scale


def quantize_sigma(sigma: torch.Tensor, bits: int) -> torch.Tensor:
    low, high = SIGMA_RANGE
    scale = _scale(bits)
    normalized = (sigma.clamp(low, high) - low) / (high - low)
    return torch.round(normalized * scale).to(torch.int32)


def dequantize_sigma(values: torch.Tensor, bits: int) -> torch.Tensor:
    low, high = SIGMA_RANGE
    scale = _scale(bits)
    return values.float() / scale * (high - low) + low


def quantize_opacity(opacity: torch.Tensor, bits: int) -> torch.Tensor:
    scale = _scale(bits)
    return torch.round(opacity.clamp(0.0, 1.0) * scale).to(torch.int32)


def dequantize_opacity(values: torch.Tensor, bits: int) -> torch.Tensor:
    scale = _scale(bits)
    return values.float() / scale
