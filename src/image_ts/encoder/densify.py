"""Triangle densification utilities."""
from __future__ import annotations

from typing import Optional

import torch

from image_ts import TriangleBatch


def _split_longest_edge(vertices: torch.Tensor, colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edges = torch.stack([
        (vertices[1] - vertices[0]).norm(p=2),
        (vertices[2] - vertices[1]).norm(p=2),
        (vertices[0] - vertices[2]).norm(p=2),
    ])
    idx = torch.argmax(edges).item()
    if idx == 0:
        a, b, c = 0, 1, 2
    elif idx == 1:
        a, b, c = 1, 2, 0
    else:
        a, b, c = 2, 0, 1
    midpoint = 0.5 * (vertices[a] + vertices[b])
    color_mid = 0.5 * (colors[a] + colors[b])
    tri1 = torch.stack([vertices[a], midpoint, vertices[c]])
    tri2 = torch.stack([midpoint, vertices[b], vertices[c]])
    col1 = torch.stack([colors[a], color_mid, colors[c]])
    col2 = torch.stack([color_mid, colors[b], colors[c]])
    return torch.stack([tri1, tri2]), torch.stack([col1, col2])


def densify_triangles(
    triangles: TriangleBatch,
    errors: torch.Tensor,
    max_triangles: int,
    importance: Optional[torch.Tensor] = None,
) -> TriangleBatch:
    current = triangles.vertices.shape[0]
    budget = max_triangles - current
    if budget <= 0:
        return triangles
    scores = errors.clone()
    if importance is not None:
        scores = scores * importance
    candidate_count = min(budget, scores.numel())
    values, indices = torch.topk(scores, k=candidate_count)
    new_vertices = []
    new_colors = []
    new_sigma = []
    new_opacity = [] if triangles.opacity is not None else None
    produced = 0
    for idx in indices.tolist():
        split_v, split_c = _split_longest_edge(triangles.vertices[idx], triangles.colors[idx])
        for tri_v, tri_c in zip(split_v, split_c):
            new_vertices.append(tri_v)
            new_colors.append(tri_c)
            new_sigma.append(triangles.sigma[idx])
            if new_opacity is not None:
                new_opacity.append(triangles.opacity[idx])
            produced += 1
            if produced >= budget:
                break
        if produced >= budget:
            break
    if not new_vertices:
        return triangles
    device = triangles.device
    vertices = torch.cat([triangles.vertices, torch.stack(new_vertices).to(device)], dim=0)
    colors = torch.cat([triangles.colors, torch.stack(new_colors).to(device)], dim=0)
    sigma = torch.cat([triangles.sigma, torch.stack(new_sigma).to(device)], dim=0)
    if triangles.opacity is not None and new_opacity is not None:
        opacity = torch.cat([triangles.opacity, torch.stack(new_opacity).to(device)], dim=0)
    else:
        opacity = triangles.opacity
    return TriangleBatch(vertices=vertices, colors=colors, sigma=sigma, opacity=opacity)
