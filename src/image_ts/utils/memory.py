"""Memory management utilities for GPU operations."""
from __future__ import annotations

import torch


# Extra safety multiplier to account for intermediate tensors beyond the main
# signed-distance buffer (barycentric coordinates, colors, weights, etc.).
MEMORY_OVERHEAD_FACTOR: float = 5.0


def get_available_memory(device: torch.device) -> float:
    """Get available GPU memory in bytes.

    Returns:
        Available memory in bytes. Returns a large value for CPU.
    """
    if device.type == "cuda":
        # Query allocator without forcing synchronization or cache flush;
        # this keeps training fast while remaining conservative via
        # MEMORY_OVERHEAD_FACTOR and safety_factor in batch sizing.
        return torch.cuda.mem_get_info(device)[0]
    # For CPU, return a large value (assume plenty of system RAM)
    return float(1e6)  # 1TB


def get_best_cuda_device() -> torch.device:
    """Select the CUDA device with the most free memory.

    Falls back to CPU if CUDA is not available or no usable device is found.
    Does not rely on NVML; uses torch.cuda.mem_get_info instead.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = None
    best_free_mem = -1

    for idx in range(torch.cuda.device_count()):
        try:
            # Query free memory on this device
            free_mem, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            # Skip devices we cannot query
            continue
        if free_mem > best_free_mem:
            best_free_mem = free_mem
            best_idx = idx

    if best_idx is None:
        # CUDA is available but mem_get_info failed for all devices
        return torch.device("cuda")

    return torch.device(f"cuda:{best_idx}")


def estimate_memory_for_signed_distance(
    num_triangles: int,
    num_points: int,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Estimate memory needed for signed_distance computation in bytes.
    
    The computation creates intermediate tensors:
    - normals_b: (T, 1, 3, 2) -> T * 1 * 3 * 2 * 4 bytes
    - offsets_b: (T, 1, 3) -> T * 1 * 3 * 4 bytes
    - points_b: (1, P, 1, 2) -> 1 * P * 1 * 2 * 4 bytes
    - dots: (T, P, 3) -> T * P * 3 * 4 bytes  (MAIN MEMORY CONSUMER)
    
    Args:
        num_triangles: Number of triangles T
        num_points: Number of evaluation points P
        dtype: Torch dtype for computation
        
    Returns:
        Estimated memory in bytes
    """
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # The dominant term is the dots tensor: (T, P, 3)
    dots_memory = num_triangles * num_points * 3 * element_size
    
    # Add some overhead for other tensors and PyTorch's internal allocations.
    return int(dots_memory * MEMORY_OVERHEAD_FACTOR)


def calculate_max_triangles_batch(
    num_points: int,
    available_memory: float,
    safety_factor: float = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> int:
    """Calculate maximum triangles to process in one batch.
    
    Args:
        num_points: Number of evaluation points
        available_memory: Available GPU memory in bytes
        safety_factor: Fraction of available memory to use. If None, uses device-specific defaults
                      (0.5 for CUDA, 0.8 for CPU) to be conservative.
        dtype: Torch dtype for computation
        device: Torch device (for determining safety factor)
        
    Returns:
        Maximum number of triangles to process in one batch
    """
    # Use device-specific safety factors if not provided
    if safety_factor is None:
        if device is not None and device.type == "cuda":
            safety_factor = 0.5  # Very conservative for CUDA (50%)
        else:
            safety_factor = 0.8  # Less conservative for CPU (80%)
    
    # Use only safety_factor of available memory
    usable_memory = available_memory * safety_factor
    
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Solve for T in: T * P * 3 * element_size * MEMORY_OVERHEAD_FACTOR <= usable_memory
    # T <= usable_memory / (P * 3 * element_size * MEMORY_OVERHEAD_FACTOR)
    max_triangles = int(
        usable_memory / (num_points * 3 * element_size * MEMORY_OVERHEAD_FACTOR)
    )
    
    # Ensure at least 1 triangle per batch
    return max(1, max_triangles)


def batch_process_triangles(
    num_triangles: int,
    num_points: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    safety_factor: float = None,
) -> tuple[int, int]:
    """Determine batch size for processing triangles within memory constraints.
    
    Args:
        num_triangles: Total number of triangles to process
        num_points: Number of evaluation points per batch
        device: Torch device (cuda or cpu)
        dtype: Torch dtype for computation
        safety_factor: Fraction of available memory to use. If None, uses device-specific defaults.
        
    Returns:
        Tuple of (batch_size, num_batches)
    """
    available_memory = get_available_memory(device)
    batch_size = calculate_max_triangles_batch(
        num_points, available_memory, safety_factor, dtype, device
    )
    
    # Don't exceed actual number of triangles
    batch_size = min(batch_size, num_triangles)
    
    num_batches = (num_triangles + batch_size - 1) // batch_size
    
    return batch_size, num_batches
