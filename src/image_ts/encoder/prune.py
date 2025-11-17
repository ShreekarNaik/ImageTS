"""Triangle pruning logic."""
from __future__ import annotations

from typing import Optional

import torch

from image_ts import TriangleBatch


def prune_triangles(
    triangles: TriangleBatch,
    scores: torch.Tensor,
    target_count: int,
    importance: Optional[torch.Tensor] = None,
) -> TriangleBatch:
    count = triangles.vertices.shape[0]
    if count <= target_count:
        return triangles
    metric = scores.clone()
    if importance is not None:
        metric = metric * importance
    keep = torch.topk(metric, k=target_count, largest=True).indices
    return triangles.select(keep)
