"""CLI to evaluate Image-TS reconstructions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_ts.codec.api import decode_triangles
from image_ts.config import RendererConfig
from image_ts.data.datasets import load_image
from image_ts.metrics import compute_metrics
from image_ts.renderer.tile_renderer import TileRenderer
from image_ts.viz.plots import plot_rd_curve, save_rd_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Image-TS reconstructions.")
    parser.add_argument("--reference", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reconstruction", type=Path)
    group.add_argument("--bitstream", type=Path)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--rd-csv", type=Path, default=None)
    parser.add_argument("--rd-plot", type=Path, default=None)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    reference = load_image(args.reference)
    reconstruction = None
    bpp = None
    if args.reconstruction is not None:
        reconstruction = load_image(args.reconstruction)
    else:
        blob = args.bitstream.read_bytes()
        header, triangles = decode_triangles(blob)
        renderer = TileRenderer(
            width=header["width"],
            height=header["height"],
            config=RendererConfig(tile_size=header["tile_size"]),
        )
        reconstruction = renderer.render(triangles)
        bpp = (len(blob) * 8) / (header["width"] * header["height"])
    metrics = compute_metrics(reconstruction, reference)
    if bpp is not None:
        metrics["bpp"] = bpp
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w") as f:
            json.dump(metrics, f, indent=2)
    if args.rd_csv is not None or args.rd_plot is not None:
        rd_point = [metrics]
        if args.rd_csv is not None:
            save_rd_data(rd_point, args.rd_csv)
        if args.rd_plot is not None:
            plot_rd_curve(rd_point, args.rd_plot)
    print(metrics)


if __name__ == "__main__":
    main()
