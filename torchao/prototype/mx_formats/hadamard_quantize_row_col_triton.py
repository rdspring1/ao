import itertools
import triton
import triton.language as tl
import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from torchao.prototype.mx_formats.hadamard_utils import (
    get_rht_matrix,
    _compute_pid,
    _nvfp4_quantize,
    _pack_fp4,
    _store_scales_non_tma,
    _swizzle_scales,
    _store_scales_swizzle,
)
from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax
from torchao.prototype.mx_formats.autotune_configs import (
    KernelConfig,
    get_best_config,
)
from torchao.utils import is_sm_at_least_100

# SM100+ autotune configs. BLOCK_M=256 enables col TMA sf store; BLOCK_N=256 enables row TMA.
HADAMARD_QUANTIZE_CONFIGS: list[KernelConfig] = [
    KernelConfig(BLOCK_M=bm, BLOCK_N=bn, NUM_STAGES=ns, NUM_WARPS=nw)
    for bm, bn, ns, nw in itertools.product(
        [128, 256],   # BLOCK_M: 256 enables columnwise TMA sf store
        [128, 256],   # BLOCK_N: >= 128 for swizzle reshape; 256 enables rowwise TMA sf store
        [2, 3, 4],    # NUM_STAGES
        [4, 8],       # NUM_WARPS
    )
]



@triton.jit
def _hadamard_quantize_row_col_kernel(
    a_desc,
    b_desc,
    c_desc,
    sf_ptr,
    rht_col_sf_desc,
    global_amax_ptr,
    rowwise_c_desc,
    rowwise_sf_ptr,
    row_sf_desc,
    rowwise_global_amax_ptr,
    seed_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr,
    SWIZZLE_SCALE_FACTORS: tl.constexpr,
    COMPUTE_ROWWISE: tl.constexpr,
):
    """Warp-specialized TMA kernel fusing RHT + NVFP4 columnwise quantization and
    optional rowwise NVFP4 quantization of the original tensor in a single pass."""
    # Persistent grid-stride loop
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    num_tiles = num_pid_m * num_pid_n

    # Load (16, 16) random hadamard matrix once
    hadamard = b_desc.load([0, 0])

    # Load global amax scalars once
    global_amax = tl.load(global_amax_ptr)
    if COMPUTE_ROWWISE:
        rowwise_global_amax = tl.load(rowwise_global_amax_ptr)

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

        # --- Columnwise path: RHT(A.t()) quantization ---

        # Transpose A_t (BLOCK_N, BLOCK_M)
        a_t = tl.trans(a)

        # Reshape to A_r (BLOCK_N * BLOCK_M // 16, 16)
        a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // 16, 16])

        # (BLOCK_N * BLOCK_M//16, 16) @ (16, 16) -> (BLOCK_N * BLOCK_M//16, 16)
        a_t_rht = tl.dot(a_t_reshape, hadamard)

        # Cast to bfloat16 like regular matmul output
        a_t_rht = a_t_rht.to(tl.bfloat16)

        # NVFP4 quantization epilogue (columnwise)
        scale_inv, scaled = _nvfp4_quantize(a_t_rht, global_amax, BLOCK_N, BLOCK_M)
        scaled_fp4x2 = _pack_fp4(
            scaled, BLOCK_N, BLOCK_M, STOCHASTIC_ROUNDING, seed_ptr, tile_id
        )

        c_desc.store([pid_n * BLOCK_N, pid_m * BLOCK_M // 2], scaled_fp4x2)

        # Store columnwise scale factors
        if SWIZZLE_SCALE_FACTORS:
            scale_inv = _swizzle_scales(scale_inv, BLOCK_N, BLOCK_M)
            if BLOCK_M >= 256:
                rht_col_sf_desc.store(
                    [pid_n * BLOCK_N // 128, pid_m * BLOCK_M // 64, 0, 0], scale_inv
                )
            else:
                _store_scales_swizzle(scale_inv, sf_ptr, pid_n, pid_m, N, M, BLOCK_N, BLOCK_M)
        else:
            if BLOCK_M >= 256:
                # TMA path: requires contiguous dim ≥ 16 bytes → float8 needs BLOCK_M ≥ 256.
                rht_col_sf_desc.store([pid_n * BLOCK_N, pid_m * BLOCK_M // 16], scale_inv)
            else:
                _store_scales_non_tma(
                    scale_inv, sf_ptr, pid_n, pid_m, M, N, BLOCK_N, BLOCK_M
                )

        # --- Rowwise path: direct quantization of A (no RHT, no transpose) ---
        if COMPUTE_ROWWISE:
            # a is (BLOCK_M, BLOCK_N) bfloat16, already loaded above.
            # _nvfp4_quantize treats first dim as "rows" and second as inner (M//16 vectors).
            # Calling with (BLOCK_M, BLOCK_N) quantizes each row of A in blocks of 16 along N.
            rowwise_scale_inv, rowwise_scaled = _nvfp4_quantize(
                a, rowwise_global_amax, BLOCK_M, BLOCK_N
            )
            # Rowwise path always uses round-to-nearest (SR degrades fwd GEMM SQNR).
            rowwise_fp4x2 = _pack_fp4(
                rowwise_scaled,
                BLOCK_M,
                BLOCK_N,
                False,
                seed_ptr,
                tile_id,
            )

            rowwise_c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N // 2], rowwise_fp4x2)

            if SWIZZLE_SCALE_FACTORS:
                rowwise_scale_inv = _swizzle_scales(rowwise_scale_inv, BLOCK_M, BLOCK_N)
                if BLOCK_N >= 256:
                    row_sf_desc.store(
                        [pid_m * BLOCK_M // 128, pid_n * BLOCK_N // 64, 0, 0],
                        rowwise_scale_inv,
                    )
                else:
                    _store_scales_swizzle(
                        rowwise_scale_inv, rowwise_sf_ptr, pid_m, pid_n, M, N, BLOCK_M, BLOCK_N
                    )
            else:
                if BLOCK_N >= 256:
                    # TMA path: inner dim = BLOCK_N//16 bytes ≥ 16 bytes for BLOCK_N ≥ 256.
                    row_sf_desc.store(
                        [pid_m * BLOCK_M, pid_n * BLOCK_N // 16], rowwise_scale_inv
                    )
                else:
                    _store_scales_non_tma(
                        rowwise_scale_inv, rowwise_sf_ptr, pid_m, pid_n, N, M, BLOCK_M, BLOCK_N,
                    )


def _launch_row_col_kernel(
    cfg: KernelConfig,
    A,
    B,
    C,
    scale_factors,
    col_global_amax,
    rowwise_C,
    rowwise_sf,
    row_global_amax,
    rand_bits,
    M: int,
    N: int,
    NUM_SMS: int,
    GROUP_SIZE_N: int,
    stochastic_rounding: bool,
    swizzle_scale_factors: bool,
    compute_rowwise: bool,
) -> None:
    """Launch _hadamard_quantize_row_col_kernel with the given KernelConfig."""
    BLOCK_M, BLOCK_N = cfg.BLOCK_M, cfg.BLOCK_N
    NUM_STAGES, NUM_WARPS = cfg.NUM_STAGES, cfg.NUM_WARPS

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_N])
    b_desc = TensorDescriptor.from_tensor(B, [16, 16])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_N, BLOCK_M // 2])

    # rht_col_sf_desc: TMA requires contiguous dim ≥ 16 bytes → float8 needs BLOCK_M ≥ 256.
    rht_col_sf_desc = None
    if BLOCK_M >= 256:
        if swizzle_scale_factors:
            rht_col_sf_desc = TensorDescriptor.from_tensor(
                scale_factors, [BLOCK_N // 128, BLOCK_M // 64, 32, 16]
            )
        else:
            rht_col_sf_desc = TensorDescriptor.from_tensor(
                scale_factors, [BLOCK_N, BLOCK_M // 16]
            )

    rowwise_c_desc = None
    row_sf_desc = None
    if compute_rowwise:
        rowwise_c_desc = TensorDescriptor.from_tensor(rowwise_C, [BLOCK_M, BLOCK_N // 2])
        if BLOCK_N >= 256:
            if swizzle_scale_factors:
                row_sf_desc = TensorDescriptor.from_tensor(
                    rowwise_sf, [BLOCK_M // 128, BLOCK_N // 64, 32, 16]
                )
            else:
                row_sf_desc = TensorDescriptor.from_tensor(
                    rowwise_sf, [BLOCK_M, BLOCK_N // 16]
                )

    _hadamard_quantize_row_col_kernel[(NUM_SMS,)](
        a_desc,
        b_desc,
        c_desc,
        scale_factors,
        rht_col_sf_desc,
        col_global_amax,
        rowwise_c_desc,
        rowwise_sf,
        row_sf_desc,
        row_global_amax,
        rand_bits,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_N=GROUP_SIZE_N,
        NUM_SMS=NUM_SMS,
        NUM_STAGES=NUM_STAGES,
        STOCHASTIC_ROUNDING=stochastic_rounding,
        SWIZZLE_SCALE_FACTORS=swizzle_scale_factors,
        COMPUTE_ROWWISE=compute_rowwise,
        num_warps=NUM_WARPS,
    )



def triton_rht_quantize_row_col(
    A: torch.Tensor,
    stochastic_rounding: bool = False,
    swizzle_scale_factors: bool = False,
    compute_rowwise: bool = True,
) -> tuple:
    """RHT + NVFP4 E2M1 columnwise quantization fused with optional rowwise quantization.

    Produces both:
      - Columnwise output: quantization of RHT(A.t()), shape (N, M//2) + (N, M//16) scales.
      - Rowwise output (when ``compute_rowwise=True``): direct NVFP4 quantization of A,
        shape (M, N//2) + (M, N//16) scales, using ``torch.max(torch.abs(A))`` as global amax.

    Both paths share the same TMA tile loads. The rowwise global amax is computed on the
    host via ``torch.max(torch.abs(A))`` before the kernel launch.

    Args:
        A: (M, N) bfloat16 tensor, row-major. M must be divisible by 16.
        stochastic_rounding (bool): Use stochastic rounding for the columnwise FP4 path.
            Stochastic rounding for wgrad GEMMs improves gradient quality via noise averaging.
            Rowwise FP4 output never uses SR — it is intended for forward GEMMs.
        swizzle_scale_factors (bool): Apply scale-factor swizzle for cuBLAS MX GEMM.
            Requires M % 128 == 0 and N % 128 == 0.
        compute_rowwise (bool): Whether to compute the rowwise quantization path.
            When False, row_fp4/row_sf/row_amax in the return tuple are None.

    Returns:
        Tuple of (col_fp4, col_sf, col_amax, row_fp4, row_sf, row_amax):
          - col_fp4:  (N, M//2) uint8 packed FP4 codes (columnwise).
          - col_sf:   (N, M//16) or swizzled float8_e4m3fn scale factors.
          - col_amax: scalar float32 global amax of RHT(A.t()).
          - row_fp4:  (M, N//2) uint8 packed FP4 codes (rowwise), or None.
          - row_sf:   (M, N//16) or swizzled float8_e4m3fn scale factors, or None.
          - row_amax: scalar float32 global amax of A, or None.

    Raises:
        NotImplementedError: If hardware is pre-SM100.
        AssertionError: If A is not bfloat16, not 2-D, not contiguous, or M % 16 != 0.

    CUDA graphs: call once before graph capture to warm up the autotuner.
    """
    if torch.cuda.is_available() and not is_sm_at_least_100():
        raise NotImplementedError(
            "Kernel requires SM100 (Blackwell); detected pre-SM100 hardware."
        )
    assert A.dtype == torch.bfloat16, f"Expected bfloat16, got {A.dtype}"
    assert A.ndim == 2, "Tensor A must be 2-D"
    assert A.is_contiguous(), "A must be row-major (contiguous)"
    assert A.shape[0] % 16 == 0, f"M must be divisible by 16, got M={A.shape[0]}"
    M, N = A.shape
    assert not compute_rowwise or N % 32 == 0, (
        f"compute_rowwise requires N % 32 == 0 (rowwise FP4 output uses TMA which "
        f"requires 16-byte-aligned inner stride; N//2 must be a multiple of 16), "
        f"got N={N}"
    )

    # Columnwise global amax: max(abs(RHT(A.t())))
    col_global_amax = triton_rht_amax(A)
    assert col_global_amax.numel() == 1
    assert col_global_amax.dtype == torch.float32

    # Rowwise global amax: max(abs(A)) — plain PyTorch, no extra kernel
    if compute_rowwise:
        row_global_amax = torch.max(torch.abs(A)).float()
    else:
        row_global_amax = torch.zeros((), dtype=torch.float32, device=A.device)

    if stochastic_rounding:
        rand_bits = torch.randint(
            low=-(2**31),
            high=2**31 - 1,
            size=(1,),
            dtype=torch.int32,
            device=A.device,
        )
    else:
        rand_bits = 0  # Safe NULL value for Triton

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    GROUP_SIZE_N: int = 8

    B = get_rht_matrix(with_random_sign_mask=True, device=A.device).to(torch.bfloat16)

    # Columnwise outputs
    C = torch.empty((N, M // 2), dtype=torch.uint8, device=A.device)

    if swizzle_scale_factors:
        assert M % 128 == 0, f"M must be divisible by 128 for swizzling, got M={M}"
        assert N % 128 == 0, f"N must be divisible by 128 for swizzling, got N={N}"
        scale_factors = torch.empty(
            (N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )
    else:
        scale_factors = torch.empty(
            (N, M // 16), dtype=torch.float8_e4m3fn, device=A.device
        )

    # Rowwise outputs
    if compute_rowwise:
        rowwise_C = torch.empty((M, N // 2), dtype=torch.uint8, device=A.device)
        if swizzle_scale_factors:
            rowwise_sf = torch.empty(
                (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
            )
        else:
            rowwise_sf = torch.empty(
                (M, N // 16), dtype=torch.float8_e4m3fn, device=A.device
            )
    else:
        # Dummy 1-element tensors; kernel constexpr COMPUTE_ROWWISE=False skips all stores.
        rowwise_C = torch.empty((1,), dtype=torch.uint8, device=A.device)
        rowwise_sf = torch.empty((1,), dtype=torch.float8_e4m3fn, device=A.device)

    # Autotune: benchmark all configs on first call for this (M, N, ...) key.
    device_cap = torch.cuda.get_device_capability(A.device)
    cache_key = ("hadamard_quantize_row_col", M, N, device_cap, compute_rowwise, swizzle_scale_factors)

    def benchmark_fn(cfg: KernelConfig) -> None:
        _launch_row_col_kernel(
            cfg, A, B, C, scale_factors, col_global_amax,
            rowwise_C, rowwise_sf, row_global_amax,
            0,  # rand_bits=0: deterministic during tuning
            M, N, NUM_SMS, GROUP_SIZE_N,
            stochastic_rounding=False,
            swizzle_scale_factors=swizzle_scale_factors,
            compute_rowwise=compute_rowwise,
        )

    cfg = get_best_config(cache_key, HADAMARD_QUANTIZE_CONFIGS, benchmark_fn)

    _launch_row_col_kernel(
        cfg,
        A,
        B,
        C,
        scale_factors,
        col_global_amax,
        rowwise_C,
        rowwise_sf,
        row_global_amax,
        rand_bits,
        M,
        N,
        NUM_SMS,
        GROUP_SIZE_N,
        stochastic_rounding,
        swizzle_scale_factors,
        compute_rowwise,
    )

    if compute_rowwise:
        return C, scale_factors, col_global_amax, rowwise_C, rowwise_sf, row_global_amax
    else:
        return C, scale_factors, col_global_amax, None, None, None
