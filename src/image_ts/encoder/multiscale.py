"""Multi-scale (coarse-to-fine) training for Image-TS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover - tqdm is optional at runtime
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from image_ts import TriangleBatch
from image_ts.config import ExperimentConfig
from image_ts.encoder.train import EncoderResult, ImageTSEncoder


def _resize_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize an image tensor (H, W, C) to (height, width, C) using bilinear filtering."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    tensor = image.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0)


def _resize_map(map_tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a single-channel map tensor (H, W) to (height, width)."""
    if map_tensor.shape[0] == height and map_tensor.shape[1] == width:
        return map_tensor
    tensor = map_tensor.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0]


def _build_pyramid(
    image: torch.Tensor,
    importance: Optional[torch.Tensor],
    levels: int,
    min_size: int,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """Build an image (and optional importance) pyramid, coarse-to-fine."""
    images: List[torch.Tensor] = [image]
    imps: List[Optional[torch.Tensor]] = [importance]
    # Repeatedly downsample by 2x until reaching desired levels or minimum size.
    while len(images) < levels:
        current = images[0]
        h, w, _ = current.shape
        if min(h, w) <= min_size * 2:
            break
        new_h = max(min_size, h // 2)
        new_w = max(min_size, w // 2)
        down_image = _resize_image(current, new_h, new_w)
        if imps[0] is not None:
            down_imp = _resize_map(imps[0], new_h, new_w)
        else:
            down_imp = None
        images.insert(0, down_image)
        imps.insert(0, down_imp)
    return images, imps


def _split_iterations(total: int, num_levels: int) -> List[int]:
    """Distribute iterations across levels with more work on finer scales."""
    if num_levels <= 0 or total <= 0:
        return []
    if num_levels == 1:
        return [total]
    # Geometric weighting: coarse -> fine gets 1,2,4,... steps.
    weights: List[int] = [2**i for i in range(num_levels)]
    weight_sum = sum(weights)
    base = [max(0, (total * w) // weight_sum) for w in weights]
    used = sum(base)
    # Distribute any remaining iterations to finest scales.
    remainder = total - used
    idx = num_levels - 1
    while remainder > 0 and idx >= 0:
        base[idx] += 1
        remainder -= 1
        idx -= 1
    return base


@dataclass
class MultiScaleResult:
    """Result of multi-scale optimization."""

    triangles: TriangleBatch
    history: List[Dict[str, float]]


def multiscale_optimize(
    image: torch.Tensor,
    config: ExperimentConfig,
    *,
    importance: Optional[torch.Tensor] = None,
    device: str = "cpu",
    levels: int = 1,
    min_size: int = 64,
    total_iterations: Optional[int] = None,
    target_loss: Optional[float] = None,
    checkpoint_interval: int = 0,
    save_intermediate_images: bool = False,
    progress_interval: int = 0,
) -> MultiScaleResult:
    """Run coarse-to-fine training over an image pyramid.

    Args:
        image: Input image tensor (H, W, C).
        config: Experiment configuration.
        importance: Optional importance map (H, W).
        device: Initial device string, e.g. "cuda" or "cpu".
        levels: Requested number of pyramid levels (1 disables multi-scale).
        min_size: Minimum min(H, W) for the coarsest level.
        total_iterations: Total iterations budget across all levels.
        target_loss: Optional loss threshold to early-stop each level.
        checkpoint_interval: Passed through to encoder.
        save_intermediate_images: Passed through to encoder.
        progress_interval: Logging interval for single-level runs (no global tqdm).

    Returns:
        MultiScaleResult with final triangles and merged history across levels.
    """
    # Single-scale fallback for compatibility.
    if levels <= 1:
        encoder = ImageTSEncoder(
            image,
            config,
            importance=importance,
            device=device,
            checkpoint_interval=checkpoint_interval,
            save_intermediate_images=save_intermediate_images,
            progress_interval=progress_interval,
        )
        result = encoder.optimize()
        return MultiScaleResult(triangles=result.triangles, history=result.history)

    # Build image / importance pyramid, coarse-to-fine.
    images, imps = _build_pyramid(image, importance, levels=levels, min_size=min_size)
    num_levels = len(images)
    # If the image is small, we might end up with fewer levels than requested.
    total_iters = int(total_iterations) if total_iterations is not None else int(
        config.encoder.schedule.iterations
    )
    per_level_iters = _split_iterations(total_iters, num_levels)
    effective_total_iters = sum(per_level_iters)

    # Global progress bar over all levels.
    global_pbar = None
    if tqdm is not None and effective_total_iters > 1:
        global_pbar = tqdm(
            total=effective_total_iters,
            desc="[image_ts] multiscale training",
            unit="iter",
        )

    from copy import deepcopy

    current_triangles: Optional[TriangleBatch] = None
    history: List[Dict[str, float]] = []
    step_offset = 0

    for level_idx, (img_level, imp_level, level_iters) in enumerate(
        zip(images, imps, per_level_iters)
    ):
        if level_iters <= 0:
            continue

        level_config = deepcopy(config)
        level_config.encoder.schedule.iterations = level_iters

        encoder = ImageTSEncoder(
            img_level,
            level_config,
            importance=imp_level,
            device=device,
            checkpoint_interval=checkpoint_interval,
            save_intermediate_images=save_intermediate_images,
            progress_interval=progress_interval,
            init_triangles=current_triangles,
        )

        level_result: EncoderResult = encoder.optimize(
            max_iterations=level_iters,
            target_loss=target_loss,
            global_progress=global_pbar,
            start_step=step_offset,
        )

        current_triangles = level_result.triangles
        # Update device for subsequent levels in case we fell back to CPU.
        device = str(current_triangles.device)

        # Annotate history with level index and merge.
        for m in level_result.history:
            m.setdefault("level", level_idx)
        history.extend(level_result.history)

        if history:
            step_offset = int(history[-1]["iteration"]) + 1

    if global_pbar is not None:
        global_pbar.close()

    if current_triangles is None:
        raise RuntimeError("Multi-scale optimization produced no triangles.")

    return MultiScaleResult(triangles=current_triangles, history=history)

