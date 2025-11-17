"""Triangle initialization using importance-aware sampling and triangulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from image_ts import TriangleBatch
from image_ts.config import EncoderConfig
from image_ts.data.importance import gradient_importance

try:
    from scipy.spatial import Delaunay  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Delaunay = None


def _sample_coordinates(importance: torch.Tensor, count: int) -> torch.Tensor:
    flat = importance.flatten()
    probs = flat / torch.sum(flat)
    idx = torch.multinomial(probs, num_samples=count, replacement=False)
    h, w = importance.shape
    ys = (idx // w).float()
    xs = (idx % w).float()
    coords = torch.stack([(xs + 0.5) / w, (ys + 0.5) / h], dim=-1)
    return coords


def _triangulate(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points for triangulation")
    if Delaunay is not None:
        simplices = Delaunay(points.cpu().numpy()).simplices
        return torch.from_numpy(simplices.astype("int64")).to(points.device)
    center = points.mean(dim=0)
    angles = torch.atan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = torch.argsort(angles)
    triangles = []
    for i in range(1, len(order) - 1):
        triangles.append(torch.tensor([order[0], order[i], order[i + 1]], dtype=torch.long))
    return torch.stack(triangles).to(points.device)


def _sample_colors(image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    h, w, c = image.shape
    x = coords[:, 0].clamp(0.0, 1.0) * (w - 1)
    y = coords[:, 1].clamp(0.0, 1.0) * (h - 1)
    x0 = torch.floor(x).long()
    x1 = torch.clamp(x0 + 1, max=w - 1)
    y0 = torch.floor(y).long()
    y1 = torch.clamp(y0 + 1, max=h - 1)
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x - x0.float()) * (y1.float() - y)
    wc = (x1.float() - x) * (y - y0.float())
    wd = (x - x0.float()) * (y - y0.float())
    image = image.to(coords.device)
    colors = (
        image[y0, x0] * wa[:, None]
        + image[y0, x1] * wb[:, None]
        + image[y1, x0] * wc[:, None]
        + image[y1, x1] * wd[:, None]
    )
    return colors


def initialize_triangles(
    image: torch.Tensor,
    config: EncoderConfig,
    importance_map: Optional[torch.Tensor] = None,
) -> TriangleBatch:
    h, w, _ = image.shape
    if importance_map is None:
        importance_map = gradient_importance(image)
    importance_map = importance_map + config.importance_smoothing
    vertex_count = max(16, config.target_triangles // 2)
    coords = _sample_coordinates(importance_map, vertex_count)
    # Add image corners for stability
    corners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=coords.device)
    coords = torch.cat([coords, corners], dim=0)
    simplices = _triangulate(coords)
    vertices = coords[simplices]
    colors = _sample_colors(image, vertices.reshape(-1, 2)).reshape(vertices.shape[0], 3, -1)
    sigma = torch.full((vertices.shape[0],), config.sigma_init, dtype=torch.float32, device=vertices.device)
    opacity = torch.ones_like(sigma)
    return TriangleBatch(vertices=vertices, colors=colors, sigma=sigma, opacity=opacity)
