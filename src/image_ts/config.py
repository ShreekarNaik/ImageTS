"""Experiment and CLI configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class RendererConfig:
    tile_size: int = 32
    topk: int = 8
    background: float = 0.0
    max_triangles_per_tile: int = 128
    epsilon: float = 1e-6


@dataclass
class LossConfig:
    lambda_l1: float = 1.0
    lambda_ms_ssim: float = 0.5
    lambda_area: float = 0.01
    lambda_sigma: float = 0.01
    importance_floor: float = 0.05


@dataclass
class EncoderSchedule:
    learning_rate: float = 5e-3
    iterations: int = 2000
    densify_every: int = 200
    prune_every: int = 400
    lr_gamma: float = 0.5
    lr_milestones: Sequence[int] = field(default_factory=lambda: [800, 1600])


@dataclass
class EncoderConfig:
    target_triangles: int = 2000
    max_triangles: int = 4000
    importance_smoothing: float = 1.5
    sigma_init: float = 0.02
    schedule: EncoderSchedule = field(default_factory=EncoderSchedule)


@dataclass
class CodecConfig:
    position_bits: int = 16
    color_bits: int = 10
    sigma_bits: int = 8
    opacity_bits: int = 6
    tile_size: int = 32


@dataclass
class ExperimentConfig:
    renderer: RendererConfig = field(default_factory=RendererConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    output_dir: Path = Path("outputs")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
