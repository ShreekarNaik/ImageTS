"""Codec APIs for Image-TS."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from image_ts import TriangleBatch
from image_ts.codec.bitstream import read_bitstream, write_bitstream
from image_ts.codec.quantization import (
    SIGMA_RANGE,
    dequantize_colors,
    dequantize_opacity,
    dequantize_positions,
    dequantize_sigma,
    quantize_colors,
    quantize_opacity,
    quantize_positions,
    quantize_sigma,
)
from image_ts.config import CodecConfig, RendererConfig
from image_ts.geometry import triangle_area
from image_ts.renderer.tile_renderer import TileRenderer


def _to_numpy_int(tensor: torch.Tensor, bits: int) -> np.ndarray:
    dtype = np.uint16 if bits <= 16 else np.uint32
    return tensor.detach().cpu().numpy().astype(dtype)


def _lod_from_count(count: int, levels: int = 4) -> List[int]:
    if count == 0:
        return [0]
    boundaries = []
    for i in range(1, levels + 1):
        boundaries.append(min(count, math.ceil(count * i / levels)))
    return sorted(set(boundaries))


import math


def encode_triangles(
    triangles: TriangleBatch,
    image_shape: Tuple[int, int, int],
    config: CodecConfig,
    importance_scores: Optional[torch.Tensor] = None,
    lod_boundaries: Optional[Sequence[int]] = None,
) -> bytes:
    if importance_scores is None:
        importance_scores = triangle_area(triangles.vertices)
    order = torch.argsort(importance_scores, descending=True)
    ordered = triangles.select(order)
    if lod_boundaries is None:
        lod_boundaries = _lod_from_count(ordered.vertices.shape[0])
    height, width, channels = image_shape
    q_pos = quantize_positions(ordered.vertices, config.position_bits)
    q_col = quantize_colors(ordered.colors, config.color_bits)
    q_sig = quantize_sigma(ordered.sigma, config.sigma_bits)
    arrays: Dict[str, np.ndarray] = {
        "positions": _to_numpy_int(q_pos, config.position_bits),
        "colors": _to_numpy_int(q_col, config.color_bits),
        "sigma": _to_numpy_int(q_sig, config.sigma_bits),
    }
    header_arrays = {
        "positions": {"shape": list(q_pos.shape), "dtype": arrays["positions"].dtype.name},
        "colors": {"shape": list(q_col.shape), "dtype": arrays["colors"].dtype.name},
        "sigma": {"shape": list(q_sig.shape), "dtype": arrays["sigma"].dtype.name},
    }
    if ordered.opacity is not None:
        q_opa = quantize_opacity(ordered.opacity, config.opacity_bits)
        arrays["opacity"] = _to_numpy_int(q_opa, config.opacity_bits)
        header_arrays["opacity"] = {"shape": list(q_opa.shape), "dtype": arrays["opacity"].dtype.name}
    header = {
        "width": width,
        "height": height,
        "channels": channels,
        "tile_size": config.tile_size,
        "bits": {
            "position": config.position_bits,
            "color": config.color_bits,
            "sigma": config.sigma_bits,
            "opacity": config.opacity_bits,
        },
        "triangle_count": ordered.vertices.shape[0],
        "lod_boundaries": list(lod_boundaries),
        "sigma_range": list(SIGMA_RANGE),
        "arrays": header_arrays,
    }
    return write_bitstream(header, arrays)


def decode_triangles(blob: bytes) -> Tuple[Dict, TriangleBatch]:
    header, arrays = read_bitstream(blob)
    bits = header["bits"]
    device = torch.device("cpu")
    vertices = dequantize_positions(torch.from_numpy(arrays["positions"]).to(device), bits["position"])
    colors = dequantize_colors(torch.from_numpy(arrays["colors"]).to(device), bits["color"])
    sigma = dequantize_sigma(torch.from_numpy(arrays["sigma"]).to(device), bits["sigma"])
    opacity = None
    if "opacity" in arrays:
        opacity = dequantize_opacity(torch.from_numpy(arrays["opacity"]).to(device), bits["opacity"])
    triangles = TriangleBatch(vertices=vertices, colors=colors, sigma=sigma, opacity=opacity)
    return header, triangles


def decode_to_image(blob: bytes, renderer_config: Optional[RendererConfig] = None, device: str = "cpu") -> torch.Tensor:
    header, triangles = decode_triangles(blob)
    renderer_config = renderer_config or RendererConfig(tile_size=header["tile_size"])
    renderer = TileRenderer(width=header["width"], height=header["height"], config=renderer_config)
    return renderer.render(triangles.to(torch.device(device)))
