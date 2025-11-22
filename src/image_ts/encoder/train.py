"""Per-image encoder for Image-TS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from image_ts import TriangleBatch
from image_ts.config import EncoderConfig, ExperimentConfig
from image_ts.encoder.densify import densify_triangles
from image_ts.encoder.init import initialize_triangles
from image_ts.encoder.prune import prune_triangles
from image_ts.losses import reconstruction_loss
from image_ts.renderer.tile_renderer import TileRenderer
from image_ts.viz.images import save_reconstruction_outputs


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
        checkpoint_interval: int = 0,
        save_intermediate_images: bool = False,
        progress_interval: int = 0,
        init_triangles: Optional[TriangleBatch] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.image = image.to(self.device)
        self.height, self.width = image.shape[:2]
        self.importance = importance.to(self.device) if importance is not None else None
        if init_triangles is not None:
            init_batch = init_triangles.to(self.device)
        else:
            init_batch = initialize_triangles(image, config.encoder, importance)
        self.params = TriangleParameters(init_batch).to(self.device)
        # Optimizer / mixed-precision state.
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.current_lr = config.encoder.schedule.learning_rate
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=self.current_lr)
        # Use the tile-based renderer for memory efficiency on large images.
        self.renderer = TileRenderer(width=self.width, height=self.height, config=config.renderer)
        self.iteration = 0
        # Checkpoint / logging helpers
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self.save_intermediate_images = bool(save_intermediate_images)
        # If progress_interval is 0, default to printing every 50 iterations for long runs.
        if progress_interval <= 0:
            self.progress_interval = 50 if config.encoder.schedule.iterations >= 200 else 10
        else:
            self.progress_interval = int(progress_interval)
        self._checkpoint_dir: Optional[Path] = None
        # Track whether we've already fallen back from CUDA to CPU due to OOM.
        self._oom_fallback_done: bool = False

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

    def _handle_oom(self, exc: RuntimeError) -> None:
        """Handle CUDA OOM by falling back to CPU exactly once."""
        if self.device.type != "cuda" or self._oom_fallback_done:
            # Either already on CPU or we already handled OOM once; re-raise.
            raise exc
        print(
            "[image_ts] CUDA out-of-memory encountered; "
            "falling back to CPU for the rest of training.",
            flush=True,
        )
        torch.cuda.empty_cache()
        with torch.no_grad():
            current = self.params.to_batch()
            cpu_batch = TriangleBatch(
                vertices=current.vertices.detach().cpu(),
                colors=current.colors.detach().cpu(),
                sigma=current.sigma.detach().cpu(),
                opacity=None if current.opacity is None else current.opacity.detach().cpu(),
            )
        self.device = torch.device("cpu")
        self.image = self.image.cpu()
        self.importance = None if self.importance is None else self.importance.cpu()
        self.params = TriangleParameters(cpu_batch).to(self.device)
        self.use_amp = False
        self.scaler = GradScaler(enabled=False)
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=self.current_lr)
        self._oom_fallback_done = True

    def _checkpoint_root(self) -> Path:
        # Lazily create the checkpoint directory under the experiment output dir.
        if self._checkpoint_dir is None:
            base_dir = getattr(self.config, "output_dir", Path("outputs"))
            self._checkpoint_dir = Path(base_dir) / "checkpoints"
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self._checkpoint_dir

    def _save_checkpoint(
        self,
        iteration: int,
        batch: TriangleBatch,
        prediction: torch.Tensor,
    ) -> None:
        ckpt_root = self._checkpoint_root()
        iter_dir = ckpt_root / f"iter_{iteration:06d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Save triangle parameters.
        triangles = TriangleBatch(
            vertices=batch.vertices.detach().cpu(),
            colors=batch.colors.detach().cpu(),
            sigma=batch.sigma.detach().cpu(),
            opacity=None if batch.opacity is None else batch.opacity.detach().cpu(),
        )
        torch.save(triangles, iter_dir / "triangles.pt")
        # Optionally save reconstruction + error map images.
        if self.save_intermediate_images:
            save_reconstruction_outputs(self.image.detach(), prediction.detach(), iter_dir)

    def optimize(
        self,
        max_iterations: Optional[int] = None,
        target_loss: Optional[float] = None,
        global_progress=None,
        start_step: int = 0,
    ) -> EncoderResult:
        schedule = self.config.encoder.schedule
        total_iters = int(max_iterations) if max_iterations is not None else int(schedule.iterations)
        history: List[Dict[str, float]] = []
        iterator = range(total_iters)
        local_pbar = None
        if global_progress is None and tqdm is not None and total_iters > 1:
            local_pbar = tqdm(iterator, desc="[image_ts] training", unit="iter")
            iterator = local_pbar

        for it in iterator:
            try:
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    batch = self.params.to_batch()
                    prediction = self.renderer.render(batch)
                    losses = reconstruction_loss(
                        prediction,
                        self.image,
                        batch,
                        self.config.losses,
                        self.importance,
                    )
                    total_loss = losses["total"]
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except RuntimeError as exc:
                # Handle CUDA out-of-memory by falling back to CPU once.
                if "out of memory" in str(exc).lower():
                    self._handle_oom(exc)
                    # Skip metrics / bookkeeping for this iteration and continue.
                    continue
                raise

            with torch.no_grad():
                metrics = {k: float(v.item()) for k, v in losses.items()}
                metrics["iteration"] = start_step + it
                history.append(metrics)
                total_loss_val = metrics.get("total", float("nan"))
                num_tris = int(batch.vertices.shape[0])

                # Progress reporting.
                if global_progress is not None:
                    global_progress.update(1)
                    global_progress.set_postfix(
                        loss=f"{total_loss_val:.6f}",
                        triangles=num_tris,
                        refresh=False,
                    )
                else:
                    if (it + 1) % self.progress_interval == 0 or (it + 1) == total_iters:
                        if local_pbar is not None:
                            local_pbar.set_postfix(loss=f"{total_loss_val:.6f}", triangles=num_tris)
                        else:
                            print(
                                f"[image_ts][iter {it + 1}/{total_iters}] "
                                f"loss={total_loss_val:.6f} triangles={num_tris}",
                                flush=True,
                            )

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
                # Periodic checkpointing of triangle parameters (and optionally images).
                if self.checkpoint_interval > 0 and (it + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(start_step + it + 1, batch, prediction)

                # Optional early stopping on loss.
                if target_loss is not None and total_loss_val <= target_loss:
                    break

        if local_pbar is not None:
            local_pbar.close()

        final_batch = self.params.to_batch()
        return EncoderResult(
            triangles=TriangleBatch(
                vertices=final_batch.vertices.detach(),
                colors=final_batch.colors.detach(),
                sigma=final_batch.sigma.detach(),
                opacity=None if final_batch.opacity is None else final_batch.opacity.detach(),
            ),
            history=history,
        )
