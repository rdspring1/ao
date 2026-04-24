# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for NVFP4 column-parallel linear (sequence-parallel TP).

Run with:
    torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py -s

Requires SM100 (Blackwell) hardware and 2 GPUs.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchao.quantization.utils import compute_error
from torchao.prototype.mx_formats.nvfp4_tensor_parallel import (
    nvfp4_col_parallel_mm,
    swap_first_dims,
)
from torchao.utils import is_sm_at_least_100


if not torch.cuda.is_available():
    pytest.skip("Requires CUDA", allow_module_level=True)

if not is_sm_at_least_100():
    pytest.skip("Requires SM100+ hardware", allow_module_level=True)


@pytest.fixture(scope="module")
def distributed_env() -> DeviceMesh:
    """Set up the 2-rank CUDA device mesh shared by all tests in this module."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip(
            "Run with: torchrun --nproc_per_node=2 -m pytest "
            "test/prototype/mx_formats/test_nvfp4_parallel.py"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, (
        f"This test requires world_size=2, got world_size={world_size}. "
        "Run with: torchrun --nproc_per_node=2 -m pytest "
        "test/prototype/mx_formats/test_nvfp4_parallel.py"
    )

    torch.manual_seed(1)
    torch.cuda.set_device(local_rank)
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
    yield device_mesh
    dist.destroy_process_group()


def test_swap_first_dims(distributed_env: DeviceMesh):
    """Verify swap_first_dims correctly de-interleaves gathered colwise tensor."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    K, M_per_rank = 128, 64
    M = M_per_rank * world_size

    # Build ground-truth [K, M//2] tensor and slice into per-rank shards [K, M_per_rank//2]
    torch.manual_seed(42)
    ground_truth = torch.randint(0, 256, (K, M // 2), dtype=torch.uint8, device=device)
    local_shard = ground_truth[
        :, rank * (M_per_rank // 2) : (rank + 1) * (M_per_rank // 2)
    ].contiguous()

    # Simulate NCCL all_gather dim-0 on local_shard [K, M_per_rank//2]
    gathered_parts = [torch.zeros_like(local_shard) for _ in range(world_size)]
    dist.all_gather(gathered_parts, local_shard)
    nccl_result = torch.cat(gathered_parts, dim=0)  # [K*W, M_per_rank//2] interleaved

    result = swap_first_dims(nccl_result, world_size)  # [K, M//2]

    assert (
        result.shape == ground_truth.shape
    ), f"Expected {ground_truth.shape}, got {result.shape}"
    torch.testing.assert_close(result, ground_truth, atol=0, rtol=0)

    # Also test 4-D scale tensor
    K_blocks = K // 128
    M_blocks_per_rank = max(1, M_per_rank // 64)
    scale_truth = torch.randint(
        0,
        256,
        (K_blocks, M_blocks_per_rank * world_size, 32, 16),
        dtype=torch.uint8,
        device=device,
    )
    scale_shard = scale_truth[
        :, rank * M_blocks_per_rank : (rank + 1) * M_blocks_per_rank, :, :
    ].contiguous()
    scale_parts = [torch.zeros_like(scale_shard) for _ in range(world_size)]
    dist.all_gather(scale_parts, scale_shard)
    scale_nccl = torch.cat(
        scale_parts, dim=0
    )  # [K_blocks*W, M_blocks_per_rank, 32, 16]
    scale_result = swap_first_dims(scale_nccl, world_size)
    assert (
        scale_result.shape == scale_truth.shape
    ), f"Expected {scale_truth.shape}, got {scale_result.shape}"
    torch.testing.assert_close(scale_result, scale_truth, atol=0, rtol=0)


def test_column_single_rank_equivalence(distributed_env: DeviceMesh):
    """Verify the TP autograd function matches the single-GPU NVFP4 path at world_size=1."""
    from torchao.prototype.mx_formats.nvfp4_linear import nvfp4_mm_triton

    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    pg = dist.new_group([0])
    M, K, N = 256, 256, 256
    if rank != 0:
        dist.barrier()
        return

    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)
    sr_seed = torch.randint(
        -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device
    )

    # Single-GPU reference
    sr_seed_ref = sr_seed.clone()
    y_ref = nvfp4_mm_triton.apply(x.clone(), w.clone(), bias.clone(), None, sr_seed_ref)

    # Column-parallel with world_size=1 (no actual distributed calls needed,
    # but we use a trivial group with just rank 0)
    sr_seed_tp = sr_seed.clone()
    y_tp = nvfp4_col_parallel_mm.apply(
        x.clone(), w.clone(), bias.clone(), sr_seed_tp, pg, 1
    )

    torch.testing.assert_close(y_ref, y_tp, atol=1e-2, rtol=1e-2)
    dist.barrier()


def test_column_forward(distributed_env: DeviceMesh):
    """Verify column-parallel forward output shape, dtype, and SQNR vs fp32 reference."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    torch.manual_seed(11)
    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_full = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias_full = torch.randn(N, dtype=torch.bfloat16, device=device)

    x_local = x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    w_local = w_full[rank * N_per_rank : (rank + 1) * N_per_rank, :]
    bias_local = bias_full[rank * N_per_rank : (rank + 1) * N_per_rank]
    sr_seed = torch.randint(
        -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device
    )

    y = nvfp4_col_parallel_mm.apply(
        x_local, w_local, bias_local, sr_seed, tp_group, world_size
    )

    assert y.shape == (
        M,
        N_per_rank,
    ), f"Rank {rank}: expected ({M}, {N_per_rank}), got {y.shape}"
    assert y.dtype == torch.bfloat16, f"Expected bfloat16, got {y.dtype}"
    assert not y.isnan().any(), "Output contains NaN"

    y_ref_full = x_full.float() @ w_full.float().t() + bias_full.float()
    y_ref_shard = y_ref_full[:, rank * N_per_rank : (rank + 1) * N_per_rank]
    sqnr = compute_error(y_ref_shard, y.float())
    SQNR_THRESHOLD = 15.0
    assert sqnr >= SQNR_THRESHOLD, f"Forward SQNR {sqnr:.2f} dB < {SQNR_THRESHOLD} dB"


def test_column_backward(distributed_env: DeviceMesh):
    """Verify column-parallel backward gradient shapes and SQNR vs fp32 reference."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    torch.manual_seed(5)
    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_full = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias_full = torch.randn(N, dtype=torch.bfloat16, device=device)
    dy_full = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    x_local = (
        x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    w_local = (
        w_full[rank * N_per_rank : (rank + 1) * N_per_rank, :]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    bias_local = (
        bias_full[rank * N_per_rank : (rank + 1) * N_per_rank]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    dy_local = dy_full[:, rank * N_per_rank : (rank + 1) * N_per_rank].contiguous()
    sr_seed = torch.randint(
        -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device
    )

    y = nvfp4_col_parallel_mm.apply(
        x_local, w_local, bias_local, sr_seed, tp_group, world_size
    )
    y.backward(dy_local)

    assert x_local.grad is not None, "x_local.grad is None"
    assert w_local.grad is not None, "w_local.grad is None"
    assert bias_local.grad is not None, "bias_local.grad is None"

    assert x_local.grad.shape == (
        M_per_rank,
        K,
    ), f"Rank {rank}: dx shape expected ({M_per_rank}, {K}), got {x_local.grad.shape}"
    assert w_local.grad.shape == (
        N_per_rank,
        K,
    ), f"Rank {rank}: dw shape expected ({N_per_rank}, {K}), got {w_local.grad.shape}"
    assert bias_local.grad.shape == (
        N_per_rank,
    ), f"Rank {rank}: db shape expected ({N_per_rank},), got {bias_local.grad.shape}"
    assert not x_local.grad.isnan().any(), "dx contains NaN"
    assert not w_local.grad.isnan().any(), "dw contains NaN"
    assert not bias_local.grad.isnan().any(), "db contains NaN"

    x_ref = x_full.float().detach().requires_grad_(True)
    w_ref = w_full.float().detach().requires_grad_(True)
    y_ref = x_ref @ w_ref.t()
    y_ref.backward(dy_full.float())

    dx_ref = x_ref.grad[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    dw_ref = w_ref.grad[rank * N_per_rank : (rank + 1) * N_per_rank, :]
    db_ref = dy_local.sum(dim=0)

    dx_sqnr = compute_error(dx_ref, x_local.grad.float())
    dw_sqnr = compute_error(dw_ref, w_local.grad.float())
    DX_SQNR_THRESHOLD = 14.0
    DW_SQNR_THRESHOLD = 14.0
    assert (
        dx_sqnr >= DX_SQNR_THRESHOLD
    ), f"dx SQNR {dx_sqnr:.2f} dB < {DX_SQNR_THRESHOLD} dB"
    assert (
        dw_sqnr >= DW_SQNR_THRESHOLD
    ), f"dw SQNR {dw_sqnr:.2f} dB < {DW_SQNR_THRESHOLD} dB"
    torch.testing.assert_close(bias_local.grad, db_ref, atol=0, rtol=0)
