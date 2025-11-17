import torch

from image_ts import TriangleBatch
from image_ts.codec.api import decode_triangles, encode_triangles
from image_ts.config import CodecConfig


def make_dummy_triangles(count: int = 4) -> TriangleBatch:
    vertices = torch.rand((count, 3, 2))
    colors = torch.rand((count, 3, 3))
    sigma = torch.rand(count) * 0.05 + 0.01
    return TriangleBatch(vertices=vertices, colors=colors, sigma=sigma)


def test_encode_decode_roundtrip():
    triangles = make_dummy_triangles()
    codec = CodecConfig()
    blob = encode_triangles(triangles, (32, 32, 3), codec)
    header, decoded = decode_triangles(blob)
    assert header["triangle_count"] == triangles.vertices.shape[0]
    assert decoded.vertices.shape == triangles.vertices.shape
