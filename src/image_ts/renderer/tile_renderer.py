"""Tile-based PyTorch renderer for Image-TS."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch

from image_ts import TriangleBatch
from image_ts.config import RendererConfig
from image_ts.geometry import barycentric_coords, edge_equations, signed_distance, triangle_bounding_box
from image_ts.window import triangle_window


@dataclass
class TileRenderer:
    width: int
    height: int
    config: RendererConfig

    def __post_init__(self) -> None:
        self.tiles_x = math.ceil(self.width / self.config.tile_size)
        self.tiles_y = math.ceil(self.height / self.config.tile_size)
        self.num_tiles = self.tiles_x * self.tiles_y

    def _normalized_coords(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def _pixel_centers(self, x0: int, y0: int, w: int, h: int, device: torch.device) -> torch.Tensor:
        ys = (torch.arange(y0, y0 + h, device=device, dtype=torch.float32) + 0.5) / self.height
        xs = (torch.arange(x0, x0 + w, device=device, dtype=torch.float32) + 0.5) / self.width
        grid = self._normalized_coords(xs, ys)
        return grid.reshape(-1, 2)

    def _build_tile_index(self, triangles: TriangleBatch) -> List[List[int]]:
        bbox = triangle_bounding_box(triangles.vertices)
        min_x = torch.clamp((bbox[:, 0] * self.width).floor().long(), 0, self.width - 1)
        min_y = torch.clamp((bbox[:, 1] * self.height).floor().long(), 0, self.height - 1)
        max_x = torch.clamp((bbox[:, 2] * self.width).ceil().long(), 0, self.width - 1)
        max_y = torch.clamp((bbox[:, 3] * self.height).ceil().long(), 0, self.height - 1)
        tile_lists: List[List[int]] = [[] for _ in range(self.num_tiles)]
        for idx in range(triangles.vertices.shape[0]):
            tx0 = (min_x[idx] // self.config.tile_size).item()
            ty0 = (min_y[idx] // self.config.tile_size).item()
            tx1 = (max_x[idx] // self.config.tile_size).item()
            ty1 = (max_y[idx] // self.config.tile_size).item()
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    if 0 <= tx < self.tiles_x and 0 <= ty < self.tiles_y:
                        tile_lists[ty * self.tiles_x + tx].append(idx)
        return tile_lists

    def render(self, triangles: TriangleBatch) -> torch.Tensor:
        device = triangles.device
        image = torch.full((self.height, self.width, triangles.colors.shape[-1]), float(self.config.background), device=device)
        weight_sums = torch.zeros((self.height, self.width), device=device)
        tile_lists = self._build_tile_index(triangles)
        normals, offsets = edge_equations(triangles.vertices)
        for tile_id, tri_indices in enumerate(tile_lists):
            if not tri_indices:
                continue
            ty, tx = divmod(tile_id, self.tiles_x)
            x0 = tx * self.config.tile_size
            y0 = ty * self.config.tile_size
            tile_w = min(self.config.tile_size, self.width - x0)
            tile_h = min(self.config.tile_size, self.height - y0)
            points = self._pixel_centers(x0, y0, tile_w, tile_h, device).unsqueeze(0)
            idx_tensor = torch.tensor(tri_indices, device=device)
            tile_vertices = triangles.vertices[idx_tensor]
            tile_colors = triangles.colors[idx_tensor]
            tile_sigma = triangles.sigma[idx_tensor]
            tile_opacity = None if triangles.opacity is None else triangles.opacity[idx_tensor]
            tile_normals = normals[idx_tensor]
            tile_offsets = offsets[idx_tensor]
            phi = signed_distance(points, tile_normals, tile_offsets)
            bary = barycentric_coords(points, tile_vertices.unsqueeze(1))
            colors = torch.sum(bary.unsqueeze(-1) * tile_colors.unsqueeze(1), dim=-2)
            weights = triangle_window(phi, tile_sigma)
            if tile_opacity is not None:
                weights = weights * tile_opacity[:, None]
            if self.config.topk > 0 and weights.shape[0] > self.config.topk:
                k = self.config.topk
                topk_vals, topk_idx = torch.topk(weights, k=k, dim=0)
                gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, colors.shape[-1])
                colors = torch.gather(colors, dim=0, index=gather_idx)
                weights = topk_vals
            blended = torch.sum(weights[..., None] * colors, dim=0)
            denom = torch.clamp(weights.sum(dim=0), min=self.config.epsilon)
            tile_pixels = (blended / denom[..., None]).reshape(tile_h, tile_w, -1)
            tile_weights = denom.reshape(tile_h, tile_w)
            image[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile_pixels
            weight_sums[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile_weights
        mask = weight_sums <= self.config.epsilon
        if mask.any():
            image[mask] = float(self.config.background)
        return image
