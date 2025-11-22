"""Test memory-aware batching in signed_distance computation."""
import torch
import pytest

from image_ts.geometry import signed_distance, edge_equations, _signed_distance_unbatched
from image_ts.utils.memory import batch_process_triangles


def test_batched_signed_distance_correctness():
    """Verify that batched and unbatched computations produce identical results."""
    num_triangles = 50
    num_points = 10000
    
    # Create sample data
    vertices = torch.rand(num_triangles, 3, 2)
    normals, offsets = edge_equations(vertices)
    points = torch.rand(1, num_points, 2)
    
    # Compute with automatic batching
    batched_result = signed_distance(points, normals, offsets)
    
    # Compute unbatched (original method)
    unbatched_result = _signed_distance_unbatched(points, normals, offsets)
    
    # Results should be identical
    assert torch.allclose(batched_result, unbatched_result, atol=1e-6)
    print(f"✓ Batched result matches unbatched for {num_triangles} triangles")


def test_batch_process_triangles_logic():
    """Verify batch size calculation."""
    device = torch.device("cpu")
    
    # For CPU, we should get very large batches (no memory constraint)
    num_triangles = 1000
    num_points = 256 * 256
    
    batch_size, num_batches = batch_process_triangles(
        num_triangles, num_points, device
    )
    
    # CPU should be able to process most/all at once
    # Just verify the math is correct
    assert batch_size >= 1, f"Batch size should be at least 1, got {batch_size}"
    assert num_batches >= 1, f"Num batches should be at least 1, got {num_batches}"
    assert batch_size * num_batches >= num_triangles, \
        f"Batch size * batches ({batch_size} * {num_batches}) should cover all triangles ({num_triangles})"
    print(f"✓ Batch processing: {num_triangles} triangles, batch_size={batch_size}, batches={num_batches}")


def test_signed_distance_shape():
    """Verify output shape is always correct."""
    for num_triangles in [10, 100, 500]:
        for num_points in [1024, 65536]:
            vertices = torch.rand(num_triangles, 3, 2)
            normals, offsets = edge_equations(vertices)
            points = torch.rand(1, num_points, 2)
            
            result = signed_distance(points, normals, offsets)
            
            assert result.shape == (num_triangles, num_points), \
                f"Expected shape ({num_triangles}, {num_points}), got {result.shape}"
    
    print(f"✓ Output shapes correct for various triangle/point counts")


if __name__ == "__main__":
    test_batched_signed_distance_correctness()
    test_batch_process_triangles_logic()
    test_signed_distance_shape()
    print("\n✅ All memory-aware batching tests passed!")
