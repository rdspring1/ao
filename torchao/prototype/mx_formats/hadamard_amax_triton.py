"""
Triton kernel for Randomized Hadamard Transform (RHT) with fused global amax reduction.

Entry point: triton_rht_amax(A) returns a scalar float32 global absolute maximum of
the post-RHT output without materializing the full (N, M) output tensor. Uses a
persistent warp-specialized TMA kernel with per-CTA cumulative max and one atomic_max
per CTA into a caller-provided scalar buffer.
"""
import itertools
import triton
import triton.language as tl
import torch
from torchao.prototype.mx_formats.hadamard_utils import get_rht_matrix, _compute_pid
from torchao.utils import is_sm_at_least_90

# SM90+ autotune configs. BLOCK_M must be divisible by 16 (RHT reshape constraint).
HADAMARD_CONFIGS: list[triton.Config] = [
    triton.Config(
        {'BLOCK_M': bm, 'BLOCK_N': bn, 'NUM_STAGES': ns},
        num_warps=nw,
        num_stages=ns,
    )
    for bm, bn, ns, nw in itertools.product(
        [64, 128],  # BLOCK_M
        [32, 64],   # BLOCK_N
        [2, 3, 4],  # NUM_STAGES
        [4, 8],     # NUM_WARPS
    )
]


@triton.autotune(configs=HADAMARD_CONFIGS, key=['M', 'N'])
@triton.jit
def _hadamard_amax_kernel(
    a_ptr,
    b_ptr,
    global_max_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Persistent RHT kernel with fused amax reduction; no output tensor written."""
    # Create TMA descriptors in-kernel from raw pointers and shape/stride
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[16, 16],
        strides=[16, 1],
        block_shape=[16, 16],
    )

    # Persistent grid-stride loop
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    num_tiles = num_pid_m * num_pid_n

    # Load (16, 16) random hadamard matrix once
    hadamard = b_desc.load([0, 0])

    # Track cumulative max across all tiles for this block
    cumulative_max = tl.zeros((BLOCK_N * BLOCK_M // 16, 16), dtype=tl.float32)

    # warp-specialized: producer warps issue TMA loads, consumer warps run wgmma
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=True,
        num_stages=NUM_STAGES,
    ):
        pid_n, pid_m = _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N)

        # Load A (BLOCK_M, BLOCK_N)
        a = a_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])

        # Transpose A_t (BLOCK_N, BLOCK_M)
        a_t = tl.trans(a)

        # Reshape to A_r (BLOCK_N * BLOCK_M//16, 16)
        a_t_r = tl.reshape(a_t, [BLOCK_N * BLOCK_M // 16, 16])

        a_t_rht = tl.dot(a_t_r, hadamard)

        # Cast to bfloat16 like regular matmul output
        a_t_rht = a_t_rht.to(tl.bfloat16)

        # Update cumulative max at tile level to avoid failing
        # TritonGPUAutomaticWarpSpecialization MLIR pass
        abs_a_t_rht = tl.abs(a_t_rht)
        cumulative_max = tl.maximum(cumulative_max, abs_a_t_rht)

    # Get scalar max for this block and update global max with atomic max operation
    tile_max = tl.max(tl.max(cumulative_max, axis=1), axis=0)
    tl.atomic_max(global_max_ptr, tile_max.to(tl.float32))


def triton_rht_amax(
    A: torch.Tensor,
    sign_vector: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Apply RHT to A and return the global absolute maximum without materializing output.

    Equivalent to rht_reference(A).abs().max().float() but fused: post-RHT values are
    never written to DRAM. Reduction is performed tile-by-tile inside the kernel and
    accumulated with a single atomic per CTA.

    Args:
        A: (M, N) bfloat16 tensor, row-major. M must be divisible by 16.

    Returns:
        Scalar float32 tensor containing max(abs(RHT(A))).

    Raises:
        NotImplementedError: If hardware is pre-SM90.
        AssertionError: If A is not bfloat16, not 2-D, not contiguous, or M % 16 != 0.

    CUDA graphs: call this function once before graph capture to warm up the autotuner.
    Subsequent calls are CUDA graph safe.
    """
    if torch.cuda.is_available() and not is_sm_at_least_90():
        raise NotImplementedError(
            "Kernel requires SM90 (Hopper); detected pre-SM90 hardware."
        )
    assert A.dtype == torch.bfloat16, f"Expected bfloat16, got {A.dtype}"
    assert A.ndim == 2, "Tensor A must be 2-D"
    assert A.is_contiguous(), "A must be row-major (contiguous)"
    assert A.shape[0] % 16 == 0, f"M must be divisible by 16, got M={A.shape[0]}"
    M, N = A.shape

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    GROUP_SIZE_N: int = 8  # L2 reuse grouping along M

    # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch space.
    # Outside torch.compile, none is set by default; mirror what torch._inductor does.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=A.device
            )
        )

    B = get_rht_matrix(sign_vector=sign_vector, device=A.device).to(torch.bfloat16)
    global_amax = torch.zeros((), dtype=torch.float32, device=A.device)

    _hadamard_amax_kernel[(NUM_SMS,)](
        A,
        B,
        global_amax,
        M,
        N,
        GROUP_SIZE_N=GROUP_SIZE_N,
        NUM_SMS=NUM_SMS,
    )
    return global_amax
