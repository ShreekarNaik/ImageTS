import torch

from image_ts import TriangleBatch
from image_ts.renderer.cpu_renderer import CPURenderer


def test_cpu_renderer_simple_triangle():
    vertices = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ])
    colors = torch.tensor([
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    sigma = torch.full((1,), 0.05)
    batch = TriangleBatch(vertices=vertices, colors=colors, sigma=sigma)
    renderer = CPURenderer(width=4, height=4)
    image = renderer.render(batch)
    assert torch.all(image[..., 0] > 0.5)
