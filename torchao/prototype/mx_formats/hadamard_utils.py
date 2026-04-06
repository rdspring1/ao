"""RHT utility functions: Hadamard matrix construction, sign vector helpers, and Triton PIDs.

Provides get_wgrad_sign_vector, get_hadamard_matrix, get_rht_matrix, cast_to_fp4x2,
and the Triton JIT helper _compute_pid.
"""

import math
import functools
import torch
import triton
import triton.language as tl


def get_wgrad_sign_vector(device) -> torch.Tensor:
    """Hard-coded random signs for Hadamard transform."""
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=device,
    )


def get_hadamard_matrix(hadamard_dimension: int, device) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix (scaled by 1/sqrt(16))."""
    assert hadamard_dimension == 16, "Only hadamard dimension 16 is supported."
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        * hadamard_scale
    )


@functools.lru_cache(maxsize=None)
def get_rht_matrix(with_random_sign_mask: bool, device) -> torch.Tensor:
    """Construct matrix used in random Hadamard transform."""
    hadamard_dimension = 16
    if with_random_sign_mask:
        signs = get_wgrad_sign_vector(device=device)
    else:
        signs = torch.ones(1, dtype=torch.float32, device=device)
    sign_matrix = signs * torch.eye(
        hadamard_dimension, dtype=torch.float32, device=device
    )
    rht_matrix = sign_matrix @ get_hadamard_matrix(hadamard_dimension, device=device)
    return rht_matrix.to(dtype=torch.bfloat16)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N: tl.constexpr):
    r"""Convert flat tile_id to (pid_n, pid_m) with L2-cache-friendly grouping."""
    group_id = tile_id // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + (tile_id % group_size_n)
    pid_m = (tile_id % num_pid_in_group) // group_size_n
    return pid_n, pid_m
