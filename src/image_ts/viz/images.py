"""Visualization helpers for reconstructions and error maps."""
from __future__ import annotations

from pathlib import Path

import torch

from image_ts.data.datasets import save_image


def save_reconstruction_outputs(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(reconstruction, output_dir / "reconstruction.png")
    error = torch.abs(reconstruction - reference).mean(dim=-1, keepdim=True)
    if torch.max(error) > 0:
        normalized = error / error.max()
    else:
        normalized = error
    error_rgb = normalized.expand(-1, -1, 3)
    save_image(error_rgb, output_dir / "error_map.png")
