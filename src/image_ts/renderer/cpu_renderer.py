"""Reference CPU renderer for Image-TS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from image_ts import TriangleBatch
from image_ts.geometry import barycentric_coords, edge_equations, signed_distance
from image_ts.window import triangle_window


@dataclass
class CPURenderer:
    width: int
    height: int
    background: float = 0.0

    def _pixel_grid(self, device: torch.device) -> torch.Tensor:
        ys = (torch.arange(self.height, device=device, dtype=torch.float32) + 0.5) / self.height
        xs = (torch.arange(self.width, device=device, dtype=torch.float32) + 0.5) / self.width
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def render(self, triangles: TriangleBatch, importance: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = triangles.device
        grid = self._pixel_grid(device)
        points = grid.reshape(-1, 2).unsqueeze(0)
        normals, offsets = edge_equations(triangles.vertices)
        phi = signed_distance(points, normals, offsets)
        bary = barycentric_coords(points, triangles.vertices.unsqueeze(1))
        vertex_colors = triangles.colors.unsqueeze(1)
        colors = torch.sum(bary.unsqueeze(-1) * vertex_colors, dim=-2)
        weights = triangle_window(phi, triangles.sigma)
        if triangles.opacity is not None:
            weights = weights * triangles.opacity[:, None]
        if importance is not None:
            imp = importance.to(device=device, dtype=weights.dtype)
            weights = weights * imp.reshape(1, -1)
        blended = torch.sum(weights[..., None] * colors, dim=0)
        denom = torch.clamp(weights.sum(dim=0), min=1e-6)
        pixels = blended / denom[..., None]
        background = torch.full_like(pixels, float(self.background))
        mask = denom <= 1e-6
        pixels = torch.where(mask[..., None], background, pixels)
        return pixels.reshape(self.height, self.width, -1)
