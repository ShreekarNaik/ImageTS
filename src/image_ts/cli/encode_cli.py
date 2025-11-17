"""CLI to encode trained triangles into a bitstream."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from image_ts.codec.api import encode_triangles
from image_ts.config import CodecConfig
from image_ts.data.datasets import load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode a TriangleBatch into an Image-TS bitstream.")
    parser.add_argument("--triangles", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Reference image for shape metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-bits", type=int, default=16)
    parser.add_argument("--color-bits", type=int, default=10)
    parser.add_argument("--sigma-bits", type=int, default=8)
    parser.add_argument("--opacity-bits", type=int, default=6)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    triangles = torch.load(args.triangles)
    image = load_image(args.image)
    codec = CodecConfig(
        position_bits=args.position_bits,
        color_bits=args.color_bits,
        sigma_bits=args.sigma_bits,
        opacity_bits=args.opacity_bits,
    )
    blob = encode_triangles(triangles, image.shape, codec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"Wrote bitstream {args.output}")


if __name__ == "__main__":
    main()
