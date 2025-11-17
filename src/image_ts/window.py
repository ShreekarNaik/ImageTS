"""Triangle splatting window functions."""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def triangle_window(phi: Tensor, sigma: Tensor, *, clamp: float = 4.0) -> Tensor:
    """Evaluate smooth window given signed distance phi and sharpness sigma."""
    sigma = torch.clamp(sigma, min=1e-4)
    while sigma.ndim < phi.ndim:
        sigma = sigma.unsqueeze(-1)
    scaled = torch.clamp(phi / sigma, min=-clamp, max=clamp)
    return torch.exp(-0.5 * scaled**2)


def blend(colors: Tensor, weights: Tensor, eps: float = 1e-6) -> Tensor:
    weighted = torch.sum(colors * weights[..., None], dim=-2)
    denom = torch.clamp(weights.sum(dim=-1, keepdim=True), min=eps)
    return weighted / denom
