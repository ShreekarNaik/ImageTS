"""Bitstream packing for Image-TS."""
from __future__ import annotations

import io
import json
import struct
from typing import Dict, Tuple

import numpy as np

MAGIC = b"IMTS"
VERSION = 1


def write_bitstream(header: Dict, arrays: Dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    buffer.write(struct.pack("<B", VERSION))
    header_bytes = json.dumps(header).encode("utf-8")
    buffer.write(struct.pack("<I", len(header_bytes)))
    buffer.write(header_bytes)
    buffer.write(struct.pack("<I", len(arrays)))
    for name, array in arrays.items():
        name_bytes = name.encode("utf-8")
        buffer.write(struct.pack("<I", len(name_bytes)))
        buffer.write(name_bytes)
        payload = array.tobytes()
        buffer.write(struct.pack("<I", len(payload)))
        buffer.write(payload)
    return buffer.getvalue()


def read_bitstream(blob: bytes) -> Tuple[Dict, Dict[str, np.ndarray]]:
    buffer = io.BytesIO(blob)
    magic = buffer.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError("Invalid bitstream magic")
    version = struct.unpack("<B", buffer.read(1))[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version {version}")
    header_len = struct.unpack("<I", buffer.read(4))[0]
    header = json.loads(buffer.read(header_len))
    num_arrays = struct.unpack("<I", buffer.read(4))[0]
    arrays: Dict[str, np.ndarray] = {}
    for _ in range(num_arrays):
        name_len = struct.unpack("<I", buffer.read(4))[0]
        name = buffer.read(name_len).decode("utf-8")
        payload_len = struct.unpack("<I", buffer.read(4))[0]
        payload = buffer.read(payload_len)
        info = header["arrays"][name]
        dtype = np.dtype(info["dtype"])
        shape = info["shape"]
        arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape)
    return header, arrays
