"""CLI for optimizing Image-TS triangles for a single image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from image_ts.codec.api import encode_triangles
from image_ts.config import ExperimentConfig
from image_ts.data.datasets import load_image
from image_ts.data.importance import load_importance
from image_ts.encoder.train import ImageTSEncoder
from image_ts.metrics import compute_metrics
from image_ts.renderer.tile_renderer import TileRenderer
from image_ts.utils.seed import seed_everything
from image_ts.viz.images import save_reconstruction_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize Image-TS triangles for an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--importance-map", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--target-triangles", type=int, default=1500)
    parser.add_argument("--max-triangles", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=8)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    seed_everything(args.seed)
    image = load_image(args.image)
    importance = None
    if args.importance_map is not None:
        importance = load_importance(args.importance_map, image.shape[:2])
    config = ExperimentConfig()
    config.encoder.target_triangles = args.target_triangles
    config.encoder.max_triangles = args.max_triangles
    config.encoder.schedule.iterations = args.iterations
    config.renderer.topk = args.topk
    config.output_dir = args.output
    config.ensure_dirs()
    encoder = ImageTSEncoder(image, config, importance=importance, device=args.device)
    result = encoder.optimize()
    triangles = result.triangles
    bitstream = encode_triangles(triangles, image.shape, config.codec)
    bit_path = args.output / "bitstream.bin"
    bit_path.write_bytes(bitstream)
    torch.save(triangles, args.output / "triangles.pt")
    renderer = TileRenderer(width=image.shape[1], height=image.shape[0], config=config.renderer)
    reconstruction = renderer.render(triangles)
    save_reconstruction_outputs(image, reconstruction, args.output)
    metrics = compute_metrics(reconstruction, image)
    metrics["bpp"] = (len(bitstream) * 8) / (image.shape[0] * image.shape[1])
    with (args.output / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (args.output / "history.json").open("w") as f:
        json.dump(result.history, f)
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
