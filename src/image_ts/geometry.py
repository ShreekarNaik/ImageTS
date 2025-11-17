"""Triangle geometry utilities."""
from __future__ import annotations

import math
from typing import Tuple

import torch

Tensor = torch.Tensor


def _cross_2d(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def triangle_area(vertices: Tensor, eps: float = 1e-8) -> Tensor:
    v1, v2, v3 = vertices.unbind(dim=-2)
    area = 0.5 * torch.abs(_cross_2d(v2 - v1, v3 - v1))
    return torch.clamp(area, min=eps)


def barycentric_coords(point: Tensor, vertices: Tensor, eps: float = 1e-8) -> Tensor:
    v1, v2, v3 = vertices.unbind(dim=-2)
    denom = _cross_2d(v2 - v1, v3 - v1)
    denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)
    w1 = _cross_2d(v2 - point, v3 - point) / denom
    w2 = _cross_2d(v3 - point, v1 - point) / denom
    w3 = 1.0 - w1 - w2
    return torch.stack([w1, w2, w3], dim=-1)


def edge_equations(vertices: Tensor) -> Tuple[Tensor, Tensor]:
    v1, v2, v3 = vertices.unbind(dim=-2)
    edges = torch.stack([v2 - v1, v3 - v2, v1 - v3], dim=-2)
    normals = torch.stack([_perp(e) for e in edges.unbind(dim=-2)], dim=-2)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    offsets = -torch.sum(normals * torch.stack([v1, v2, v3], dim=-2), dim=-1)
    return normals, offsets


def _perp(vec: Tensor) -> Tensor:
    return torch.stack([-vec[..., 1], vec[..., 0]], dim=-1)


def signed_distance(points: Tensor, normals: Tensor, offsets: Tensor) -> Tensor:
    \"\"\"Evaluate Triangle SDF for a broadcastable grid of points.\"\"\"\n+    normals_b = normals.unsqueeze(-3)\n+    offsets_b = offsets.unsqueeze(-3)\n+    points_b = points.unsqueeze(-2)\n+    dots = torch.sum(normals_b * points_b, dim=-1) + offsets_b\n+    return torch.amax(dots, dim=-1)\n*** End Patch


def triangle_incenter(vertices: Tensor, eps: float = 1e-8) -> Tensor:
    v1, v2, v3 = vertices.unbind(dim=-2)
    a = torch.norm(v2 - v3, dim=-1)
    b = torch.norm(v3 - v1, dim=-1)
    c = torch.norm(v1 - v2, dim=-1)
    perimeter = a + b + c + eps
    return (a[..., None] * v1 + b[..., None] * v2 + c[..., None] * v3) / perimeter[..., None]


def clamp_vertices(vertices: Tensor, min_val: float = 0.0, max_val: float = 1.0) -> Tensor:
    return vertices.clamp(min=min_val, max=max_val)


def triangle_bounding_box(vertices: Tensor) -> Tensor:
    mins = vertices.amin(dim=-2)
    maxs = vertices.amax(dim=-2)
    return torch.cat([mins, maxs], dim=-1)
