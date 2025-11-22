"""Triangle geometry utilities."""
from __future__ import annotations

import math
from typing import Tuple
import warnings

import torch

from image_ts.utils.memory import batch_process_triangles

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
    """Evaluate Triangle SDF for a broadcastable grid of points with automatic batching.
    
    Automatically batches processing based on available GPU memory to prevent OOM errors.

    Args:
        points: (1, P, 2) or (B, P, 2) - evaluation points
        normals: (T, 3, 2) - edge normals for triangles
        offsets: (T, 3) - edge offsets for triangles

    Returns: (T, P) signed distances using max over half-space edge equations.
    """
    num_triangles = normals.shape[0]
    num_points = points.shape[1]
    device = normals.device
    dtype = normals.dtype
    
    # Check if we need batching
    batch_size, num_batches = batch_process_triangles(
        num_triangles, num_points, device, dtype
    )
    
    # If all triangles fit in one batch, use original computation
    if num_batches == 1:
        return _signed_distance_unbatched(points, normals, offsets)
    
    # Log when batching is needed (for debugging)
    warnings.warn(
        f"Processing {num_triangles} triangles in {num_batches} batches "
        f"(batch_size={batch_size}) to fit in GPU memory",
        RuntimeWarning,
        stacklevel=2
    )
    
    # Otherwise, batch the computation
    results = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_triangles)
        
        batch_normals = normals[start_idx:end_idx]
        batch_offsets = offsets[start_idx:end_idx]

        batch_result = _signed_distance_unbatched(points, batch_normals, batch_offsets)
        results.append(batch_result)

        # Drop references so Python can reclaim memory promptly.
        del batch_result, batch_normals, batch_offsets
    
    # Concatenate results along triangle dimension
    return torch.cat(results, dim=0)  # (T, P)


def _signed_distance_unbatched(points: Tensor, normals: Tensor, offsets: Tensor) -> Tensor:
    """Single-batch signed distance computation (original implementation).

    points: (1, P, 2) or (B, P, 2)
    normals: (T, 3, 2)
    offsets: (T, 3)

    Returns: (T, P) signed distances using max over half-space edge equations.
    """
    normals_b = normals.unsqueeze(1)      # (T, 1, 3, 2)
    offsets_b = offsets.unsqueeze(1)      # (T, 1, 3)
    points_b = points.unsqueeze(-2)       # (1 or B, P, 1, 2) -> broadcast over T
    dots = torch.sum(normals_b * points_b, dim=-1) + offsets_b  # (T, P, 3)
    return torch.amax(dots, dim=-1)       # (T, P)



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
