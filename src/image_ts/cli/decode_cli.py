"""CLI to decode Image-TS bitstreams."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from image_ts.codec.api import decode_triangles
from image_ts.config import RendererConfig
from image_ts.data.datasets import save_image
from image_ts.renderer.tile_renderer import TileRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode an Image-TS bitstream to an image.")
    parser.add_argument("--bitstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-triangles", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=8)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    blob = args.bitstream.read_bytes()
    header, triangles = decode_triangles(blob)
    renderer_cfg = RendererConfig(tile_size=header["tile_size"], topk=args.topk)
    renderer = TileRenderer(width=header["width"], height=header["height"], config=renderer_cfg)
    reconstruction = renderer.render(triangles, show_progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(reconstruction, args.output)
    if args.save_triangles is not None:
        torch.save(triangles, args.save_triangles)
    print(f"Decoded image saved to {args.output}")


if __name__ == "__main__":
    main()
