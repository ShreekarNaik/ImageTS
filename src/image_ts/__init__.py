"""Image-TS: Triangle-based image compression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "TriangleBatch",
    "ensure_tensor",
]


def ensure_tensor(value: torch.Tensor | float | int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert scalars to tensors while keeping tensors untouched."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device) if device is not None else value
    return torch.as_tensor(value, device=device)


@dataclass
class TriangleBatch:
    """Container storing triangle parameters for rendering/optimization."""

    vertices: torch.Tensor  # (T, 3, 2)
    colors: torch.Tensor  # (T, 3, C)
    sigma: torch.Tensor  # (T,)
    opacity: Optional[torch.Tensor] = None  # (T,)

    def __post_init__(self) -> None:
        if self.vertices.ndim != 3 or self.vertices.shape[1:] != (3, 2):
            raise ValueError("vertices must have shape (T, 3, 2)")
        if self.colors.ndim != 3 or self.colors.shape[:2] != self.vertices.shape[:2]:
            raise ValueError("colors must have shape (T, 3, C) and share triangle count")
        if self.sigma.ndim != 1 or self.sigma.shape[0] != self.vertices.shape[0]:
            raise ValueError("sigma must have shape (T,)")
        if self.opacity is not None and (
            self.opacity.ndim != 1 or self.opacity.shape[0] != self.vertices.shape[0]
        ):
            raise ValueError("opacity must have shape (T,)")

    @property
    def device(self) -> torch.device:
        return self.vertices.device

    def to(self, device: torch.device) -> "TriangleBatch":
        return TriangleBatch(
            vertices=self.vertices.to(device),
            colors=self.colors.to(device),
            sigma=self.sigma.to(device),
            opacity=None if self.opacity is None else self.opacity.to(device),
        )

    def clone(self) -> "TriangleBatch":
        return TriangleBatch(
            vertices=self.vertices.clone(),
            colors=self.colors.clone(),
            sigma=self.sigma.clone(),
            opacity=None if self.opacity is None else self.opacity.clone(),
        )

    def select(self, indices: torch.Tensor) -> "TriangleBatch":
        idx = indices.to(self.vertices.device)
        return TriangleBatch(
            vertices=self.vertices[idx],
            colors=self.colors[idx],
            sigma=self.sigma[idx],
            opacity=None if self.opacity is None else self.opacity[idx],
        )
