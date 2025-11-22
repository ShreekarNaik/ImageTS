"""Tile-based PyTorch renderer for Image-TS."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from image_ts import TriangleBatch
from image_ts.config import RendererConfig
from image_ts.geometry import (
    barycentric_coords,
    edge_equations,
    _signed_distance_unbatched,
    triangle_bounding_box,
)
from image_ts.utils.memory import batch_process_triangles
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

    def render(self, triangles: TriangleBatch, show_progress: bool = False) -> torch.Tensor:
        device = triangles.device
        image = torch.full((self.height, self.width, triangles.colors.shape[-1]), float(self.config.background), device=device)
        weight_sums = torch.zeros((self.height, self.width), device=device)
        tile_lists = self._build_tile_index(triangles)
        normals, offsets = edge_equations(triangles.vertices)
        total_tiles = len(tile_lists)

        tile_iter = range(total_tiles)
        pbar = None
        if show_progress and tqdm is not None and total_tiles > 0:
            pbar = tqdm(range(total_tiles), desc="[image_ts] render tiles", unit="tile")
            tile_iter = pbar

        for tile_id in tile_iter:
            tri_indices = tile_lists[tile_id]
            if not tri_indices:
                continue
            if show_progress and (tqdm is None):
                # Fallback lightweight render progress indicator without tqdm.
                processed = tile_id + 1
                step = max(1, total_tiles // 10)
                if processed == 1 or processed % step == 0 or processed == total_tiles:
                    print(
                        f"[image_ts][render] tiles {processed}/{total_tiles}",
                        end="\r",
                        flush=True,
                    )
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

            num_triangles = tile_vertices.shape[0]
            num_points = points.shape[1]
            channels = tile_colors.shape[-1]

            batch_size, num_batches = batch_process_triangles(
                num_triangles=num_triangles,
                num_points=num_points,
                device=device,
                dtype=tile_vertices.dtype,
            )

            if self.config.topk > 0:
                k = self.config.topk
                best_weights = torch.zeros((k, num_points), device=device, dtype=tile_vertices.dtype)
                best_colors = torch.zeros((k, num_points, channels), device=device, dtype=tile_colors.dtype)
            else:
                blended = torch.zeros((num_points, channels), device=device, dtype=tile_colors.dtype)
                denom = torch.zeros((num_points,), device=device, dtype=tile_vertices.dtype)

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_triangles)
                if start >= end:
                    break

                b_vertices = tile_vertices[start:end]
                b_colors = tile_colors[start:end]
                b_sigma = tile_sigma[start:end]
                b_normals = tile_normals[start:end]
                b_offsets = tile_offsets[start:end]
                b_opacity = None if tile_opacity is None else tile_opacity[start:end]

                phi = _signed_distance_unbatched(points, b_normals, b_offsets)
                bary = barycentric_coords(points, b_vertices.unsqueeze(1))
                colors = torch.sum(bary.unsqueeze(-1) * b_colors.unsqueeze(1), dim=-2)
                weights = triangle_window(phi, b_sigma)
                if b_opacity is not None:
                    weights = weights * b_opacity[:, None]

                if self.config.topk > 0:
                    all_weights = torch.cat([best_weights, weights], dim=0)
                    topk_vals, topk_idx = torch.topk(all_weights, k=k, dim=0)
                    all_colors = torch.cat([best_colors, colors], dim=0)
                    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, channels)
                    best_colors = torch.gather(all_colors, dim=0, index=gather_idx)
                    best_weights = topk_vals
                else:
                    blended = blended + torch.sum(weights[..., None] * colors, dim=0)
                    denom = denom + weights.sum(dim=0)

                # Drop local references so Python can reclaim memory promptly.
                del phi, bary, colors, weights, b_vertices, b_colors, b_sigma, b_normals, b_offsets, b_opacity

            if self.config.topk > 0:
                blended = torch.sum(best_weights[..., None] * best_colors, dim=0)
                denom = torch.clamp(best_weights.sum(dim=0), min=self.config.epsilon)
            else:
                denom = torch.clamp(denom, min=self.config.epsilon)

            tile_pixels = (blended / denom[..., None]).reshape(tile_h, tile_w, -1)
            tile_weights = denom.reshape(tile_h, tile_w)
            image[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile_pixels
            weight_sums[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile_weights
        if show_progress and tqdm is None and total_tiles > 0:
            # Finish the fallback progress line.
            print()
        mask = weight_sums <= self.config.epsilon
        if mask.any():
            image[mask] = float(self.config.background)
        return image
