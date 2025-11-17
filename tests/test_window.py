import torch

from image_ts.window import triangle_window


def test_triangle_window_monotonic():
    phi = torch.tensor([[0.0, 0.5, 1.0]])
    sigma = torch.tensor([0.1])
    values = triangle_window(phi, sigma)
    assert values[0, 0] >= values[0, 1] >= values[0, 2]
