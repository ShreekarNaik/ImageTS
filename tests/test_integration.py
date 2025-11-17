import torch

from image_ts import TriangleBatch
from image_ts.codec.api import decode_to_image, encode_triangles
from image_ts.config import CodecConfig


def test_encode_decode_image():
    image_shape = (8, 8, 3)
    vertices = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])
    colors = torch.tensor([
        [[1.0, 0.0, 0.0]] * 3,
        [[0.0, 0.0, 1.0]] * 3,
    ])
    sigma = torch.full((2,), 0.05)
    batch = TriangleBatch(vertices=vertices, colors=colors, sigma=sigma)
    blob = encode_triangles(batch, image_shape, CodecConfig())
    reconstruction = decode_to_image(blob)
    assert reconstruction.shape == image_shape
