"""Plotting utilities for Image-TS RD curves."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def save_rd_data(points: Iterable[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    points = list(points)
    if not points:
        return
    fieldnames = sorted(points[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def plot_rd_curve(points: Iterable[dict], plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points = list(points)
    if not points:
        return
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    bpp = [p.get("bpp", 0.0) for p in points]
    psnr = [p.get("psnr", 0.0) for p in points]
    plt.figure()
    plt.plot(bpp, psnr, marker="o")
    plt.xlabel("bpp")
    plt.ylabel("PSNR")
    plt.title("Image-TS RD Curve")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
