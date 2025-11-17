import torch

from image_ts.geometry import barycentric_coords, triangle_area


def test_barycentric_sum_to_one():
    vertices = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ])
    points = torch.tensor([[[0.25, 0.25]]])
    bary = barycentric_coords(points, vertices.unsqueeze(1))
    assert torch.allclose(bary.sum(dim=-1), torch.ones_like(bary[..., 0]))


def test_triangle_area_positive():
    vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    area = triangle_area(vertices)
    assert torch.all(area > 0)
