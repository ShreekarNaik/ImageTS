"""Per-image encoder for Image-TS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from image_ts import TriangleBatch
from image_ts.config import EncoderConfig, ExperimentConfig
from image_ts.encoder.densify import densify_triangles
from image_ts.encoder.init import initialize_triangles
from image_ts.encoder.prune import prune_triangles
from image_ts.losses import reconstruction_loss
from image_ts.renderer.cpu_renderer import CPURenderer


def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def _inv_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.exp(x) - 1 + eps)


class TriangleParameters(torch.nn.Module):
    def __init__(self, batch: TriangleBatch):
        super().__init__()
        self.vertex_logit = torch.nn.Parameter(_logit(batch.vertices))
        self.color_logit = torch.nn.Parameter(_logit(batch.colors))
        self.sigma_unconstrained = torch.nn.Parameter(_inv_softplus(batch.sigma))
        if batch.opacity is not None:
            self.opacity_logit = torch.nn.Parameter(_logit(batch.opacity))
        else:
            self.opacity_logit = None

    def to_batch(self) -> TriangleBatch:
        vertices = torch.sigmoid(self.vertex_logit)
        colors = torch.sigmoid(self.color_logit)
        sigma = torch.nn.functional.softplus(self.sigma_unconstrained)
        opacity = torch.sigmoid(self.opacity_logit) if self.opacity_logit is not None else None
        return TriangleBatch(vertices=vertices, colors=colors, sigma=sigma, opacity=opacity)


@dataclass
class EncoderResult:
    triangles: TriangleBatch
    history: List[Dict[str, float]]


class ImageTSEncoder:
    def __init__(
        self,
        image: torch.Tensor,
        config: ExperimentConfig,
        importance: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.image = image.to(self.device)
        self.height, self.width = image.shape[:2]
        self.importance = importance.to(self.device) if importance is not None else None
        init_batch = initialize_triangles(image, config.encoder, importance)
        self.params = TriangleParameters(init_batch).to(self.device)
        self.current_lr = config.encoder.schedule.learning_rate
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=self.current_lr)
        self.renderer = CPURenderer(width=self.width, height=self.height)
        self.iteration = 0

    def _triangle_importance(self, triangles: TriangleBatch) -> Optional[torch.Tensor]:
        if self.importance is None:
            return None
        centers = triangles.vertices.mean(dim=1)
        xs = (centers[:, 0] * (self.width - 1)).long().clamp(0, self.width - 1)
        ys = (centers[:, 1] * (self.height - 1)).long().clamp(0, self.height - 1)
        return self.importance[ys, xs]

    def _triangle_errors(self, prediction: torch.Tensor, triangles: TriangleBatch) -> torch.Tensor:
        err_map = torch.abs(prediction - self.image).mean(dim=-1)
        centers = triangles.vertices.mean(dim=1)
        xs = (centers[:, 0] * (self.width - 1)).long().clamp(0, self.width - 1)
        ys = (centers[:, 1] * (self.height - 1)).long().clamp(0, self.height - 1)
        return err_map[ys, xs]

    def _maybe_adjust_lr(self, iteration: int) -> None:
        schedule = self.config.encoder.schedule
        if iteration in schedule.lr_milestones:
            self.current_lr *= schedule.lr_gamma
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr

    def _reinitialize(self, batch: TriangleBatch) -> None:
        self.params = TriangleParameters(batch).to(self.device)
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=self.current_lr)

    def optimize(self) -> EncoderResult:
        schedule = self.config.encoder.schedule
        history: List[Dict[str, float]] = []
        for it in range(schedule.iterations):
            self.optimizer.zero_grad()
            batch = self.params.to_batch()
            prediction = self.renderer.render(batch)
            losses = reconstruction_loss(prediction, self.image, batch, self.config.losses, self.importance)
            losses["total"].backward()
            self.optimizer.step()
            with torch.no_grad():
                metrics = {k: float(v.item()) for k, v in losses.items()}
                metrics["iteration"] = it
                history.append(metrics)
                tri_errors = self._triangle_errors(prediction, batch)
                tri_importance = self._triangle_importance(batch)
                if (it + 1) % schedule.densify_every == 0:
                    new_batch = densify_triangles(
                        TriangleBatch(
                            vertices=batch.vertices.detach(),
                            colors=batch.colors.detach(),
                            sigma=batch.sigma.detach(),
                            opacity=None if batch.opacity is None else batch.opacity.detach(),
                        ),
                        tri_errors.detach(),
                        self.config.encoder.max_triangles,
                        tri_importance.detach() if tri_importance is not None else None,
                    )
                    if new_batch.vertices.shape[0] != batch.vertices.shape[0]:
                        self._reinitialize(new_batch)
                        batch = self.params.to_batch()
                if (it + 1) % schedule.prune_every == 0:
                    target = max(self.config.encoder.target_triangles, batch.vertices.shape[0] // 2)
                    pruned = prune_triangles(
                        TriangleBatch(
                            vertices=batch.vertices.detach(),
                            colors=batch.colors.detach(),
                            sigma=batch.sigma.detach(),
                            opacity=None if batch.opacity is None else batch.opacity.detach(),
                        ),
                        tri_errors.detach(),
                        target,
                        tri_importance.detach() if tri_importance is not None else None,
                    )
                    if pruned.vertices.shape[0] != batch.vertices.shape[0]:
                        self._reinitialize(pruned)
                self._maybe_adjust_lr(it + 1)
        final_batch = self.params.to_batch()
        return EncoderResult(triangles=TriangleBatch(
            vertices=final_batch.vertices.detach(),
            colors=final_batch.colors.detach(),
            sigma=final_batch.sigma.detach(),
            opacity=None if final_batch.opacity is None else final_batch.opacity.detach(),
        ), history=history)
