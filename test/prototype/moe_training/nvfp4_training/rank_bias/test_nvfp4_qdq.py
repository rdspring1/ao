"""Bitwise-equivalence tests: CuTe DSL kernels vs PyTorch oracle vs kitchen math.

Run with ``pytest -q test_nvfp4_qdq.py`` on any CUDA machine; nothing here needs
kitchen, psx_formats or TransformerEngine.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_cutedsl as ct
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_reference as ref
from torchao.prototype.moe_training.nvfp4_training.rank_bias.analyze_rank_bias import make_rank_labels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

SHAPES = [
    (64, 64),  # no padding on either path
    (130, 48),  # pads rows (32/128) and cols (64) on both paths
    (1, 16),  # single block
    (512, 1024),  # large enough to hit rounding boundaries
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype]
    return t.contiguous().view(dtype)


def _sample(shape, seed=0, scale=3.0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * scale
    ).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# CuTe DSL == PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_qdq_bitwise_matches_reference(shape, transpose, use_sr):
    x = _sample(shape)
    got = ct.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=11)
    want = ref.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=11)
    assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_quantize_intermediates_bitwise_match(transpose, use_sr):
    """Codes, block scales and the global scale must match, not just the product."""
    x = _sample((512, 1024), seed=5)
    got = ct.quantize(x, transpose=transpose, use_sr=use_sr, seed=3)
    want = ref.quantize(x, transpose=transpose, use_sr=use_sr, seed=3)
    assert torch.equal(_bits(got.data_q), _bits(want.data_q))
    assert torch.equal(_bits(got.block_descale), _bits(want.block_descale))
    assert torch.equal(_bits(got.global_descale), _bits(want.global_descale))


def test_sr_seeds_decorrelate_and_average_out():
    """SR must vary per seed and its mean must converge on the input."""
    x = _sample((256, 256), seed=9)
    a = ct.quant_dequant(x, use_sr=True, seed=1)
    b = ct.quant_dequant(x, use_sr=True, seed=2)
    assert not torch.equal(a, b)
    mean = torch.zeros_like(x, dtype=torch.float32)
    for seed in range(64):
        mean += ct.quant_dequant(x, use_sr=True, seed=seed).float()
    mean /= 64
    one_trial_err = (a.float() - x.float()).square().mean()
    assert (mean - x.float()).square().mean() < one_trial_err / 8


def test_degenerate_blocks():
    """Zeros (incl. -0.0), a zero block, and a block that saturates FP4."""
    x = torch.zeros(32, 32, device="cuda", dtype=torch.bfloat16)
    x[0, :16] = torch.tensor([0.0, -0.0] * 8, device="cuda", dtype=torch.bfloat16)
    x[1, :16] = 1e3  # amax block: every element saturates to the FP4 max
    x[2, 0] = 1e-4  # tiny value against a huge tensor amax -> block scale flushes
    for transpose in (False, True):
        for use_sr in (False, True):
            got = ct.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=4)
            want = ref.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=4)
            assert torch.equal(_bits(got), _bits(want)), (transpose, use_sr)


def test_all_zero_tensor():
    """amax == 0 makes the global scale overflow; kitchen falls back to unit scale."""
    x = torch.zeros(32, 32, device="cuda", dtype=torch.bfloat16)
    for transpose in (False, True):
        got = ct.quant_dequant(x, transpose=transpose, use_sr=True, seed=1)
        want = ref.quant_dequant(x, transpose=transpose, use_sr=True, seed=1)
        assert torch.equal(_bits(got), _bits(want))
        assert torch.equal(got.float(), torch.zeros_like(got, dtype=torch.float32))


# ---------------------------------------------------------------------------
# E4M3 block-scale rounding == torch's float8_e4m3fn cast
# ---------------------------------------------------------------------------


@cute.kernel
def _e4m3_probe_kernel(gX: cute.Tensor, gO: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < n:
        gO[i] = ct._e4m3_rne(cutlass.Float32(gX[i]))


@cute.jit
def _e4m3_probe(mX: cute.Tensor, mO: cute.Tensor, n: cutlass.Int32, stream):
    _e4m3_probe_kernel(mX, mO, n).launch(
        grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )


def test_e4m3_rne_matches_torch_cast_exhaustively():
    """Every fp32 bit pattern in [0, 449) plus the saturating tail."""
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    chunk = 1 << 22
    for start in range(0, 0x43E10000, chunk):
        pattern = torch.arange(
            start, min(start + chunk, 0x43E10000), device="cuda", dtype=torch.int64
        ).to(torch.int32)
        vals = pattern.view(torch.float32)
        out = torch.empty_like(vals)
        _e4m3_probe(from_dlpack(vals), from_dlpack(out), vals.numel(), stream)
        want = vals.to(torch.float8_e4m3fn).float()
        assert torch.equal(_bits(out), _bits(want)), f"chunk at {start:#x}"

    tail = torch.tensor(
        [448.0, 448.001, 449.0, 464.0, 1e10, float("inf")], device="cuda"
    )
    out = torch.empty_like(tail)
    _e4m3_probe(from_dlpack(tail), from_dlpack(out), tail.numel(), stream)
    assert torch.equal(_bits(out), _bits(tail.to(torch.float8_e4m3fn).float()))


# ---------------------------------------------------------------------------
# PyTorch reference == kitchen's nvfp_utils / cast_utils source
# ---------------------------------------------------------------------------


def _kitchen_cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Verbatim kitchen/cast_utils.py::cast_to_fp4_e2m1."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _kitchen_cast_to_fp4_e2m1_sr(x: torch.Tensor, sampled_prob: torch.Tensor):
    """kitchen/cast_utils.py::cast_to_fp4_e2m1_sr; only its ``torch.rand`` call
    is replaced by the injected uniforms."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x_fp4_high = torch.where(
        x > 4,
        6,
        torch.where(
            x > 3,
            4,
            torch.where(
                x > 2,
                3,
                torch.where(
                    x > 1.5, 2, torch.where(x > 1.0, 1.5, torch.where(x > 0.5, 1, 0.5))
                ),
            ),
        ),
    )
    x_fp4_low = torch.where(
        x > 4,
        4,
        torch.where(
            x > 3,
            3,
            torch.where(
                x > 2,
                2,
                torch.where(
                    x > 1.5,
                    1.5,
                    torch.where(x > 1.0, 1.0, torch.where(x > 0.5, 0.5, 0.0)),
                ),
            ),
        ),
    )
    prob_up = (x - x_fp4_low) / (x_fp4_high - x_fp4_low)
    return torch.where(sampled_prob < prob_up, x_fp4_high, x_fp4_low) * sign


def _kitchen_qdq(data_hp: torch.Tensor, use_sr: bool, uniforms=None) -> torch.Tensor:
    """kitchen/nvfp_utils.py::to_nvfp_verbose + from_nvfp_verbose, transcribed for
    quant_tile_shape=(1, 16), E2M1, E4M3_RNE scales, per-tensor 2-level scaling."""
    M, N = data_hp.shape
    blk_m, blk_n, block_size0, block_size1 = M, N // 16, 1, 16
    amax_encoded_data, amax_encoded_block_scale = 6.0, 448.0

    data_blocked = (
        data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
        .permute(0, 2, 1, 3)
        .contiguous()
        .flatten(start_dim=2)
    )
    max_abs_per_block = torch.amax(
        torch.abs(data_blocked), dim=-1, keepdim=True
    ).float()

    global_amax = data_hp.abs().float().amax().reshape(1)  # compute_tensor_absmax
    global_scaling_factor = torch.div(
        amax_encoded_data * amax_encoded_block_scale, global_amax
    )
    global_descaling_factor = torch.reciprocal(global_scaling_factor)
    overflow_mask = torch.isinf(global_scaling_factor)
    if torch.any(overflow_mask):
        ones = torch.ones_like(global_scaling_factor)
        global_scaling_factor = torch.where(overflow_mask, ones, global_scaling_factor)
        global_descaling_factor = torch.where(
            overflow_mask, ones, global_descaling_factor
        )
    global_scaling_factor = global_scaling_factor.expand(blk_m, blk_n, 1)
    global_descaling_factor = global_descaling_factor.expand(blk_m, blk_n, 1)

    block_descaling_factor_e4m3 = (
        torch.div(max_abs_per_block, amax_encoded_data) * global_scaling_factor
    ).to(torch.float8_e4m3fn)
    block_scaling_factor = torch.where(
        block_descaling_factor_e4m3 == 0,
        1.0,
        torch.reciprocal(block_descaling_factor_e4m3.float() * global_descaling_factor),
    )

    data_hp_q = data_blocked * block_scaling_factor
    if use_sr:
        data_hp_q = _kitchen_cast_to_fp4_e2m1_sr(
            data_hp_q, uniforms.reshape(data_hp_q.shape)
        )
    else:
        data_hp_q = _kitchen_cast_to_fp4_e2m1(data_hp_q)

    data_hp_q = (
        data_hp_q.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(M, N)
        .float()
    )
    block_descale = block_descaling_factor_e4m3.squeeze(-1)
    global_descale = global_descaling_factor.squeeze(-1)

    # from_nvfp_verbose
    restored = (
        data_hp_q
        * block_descale.float()
        .repeat_interleave(block_size0, 0)
        .repeat_interleave(block_size1, 1)
        * global_descale.repeat_interleave(block_size0, 0).repeat_interleave(
            block_size1, 1
        )
    )
    return restored.to(torch.bfloat16)  # QuantizeOpBase.dequantize


@pytest.mark.parametrize("use_sr", [False, True])
def test_reference_matches_kitchen_source(use_sr):
    """The oracle reproduces kitchen's own reference math bit for bit."""
    x = _sample((256, 512), seed=7)
    uniforms = ref.philox_uniform(x.numel(), 21, x.device) if use_sr else None
    want = _kitchen_qdq(x, use_sr, uniforms)
    got = ref.quant_dequant(x, use_sr=use_sr, seed=21)
    assert torch.equal(_bits(got), _bits(want))


# ---------------------------------------------------------------------------
# Rank labelling used by the sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transpose", [False, True])
def test_rank_labels_follow_the_quantization_block_axis(transpose):
    """Identity ranks within a row (1xB); transpose ranks within a column (Bx1)."""
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 0.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
    )
    labels = make_rank_labels(x, block_size=2, transpose=transpose)
    expected = (
        [[2, 1], [1, 0], [2, 2], [1, 1]]
        if transpose
        else [[2, 1], [1, 0], [2, 1], [2, 1]]
    )
    assert labels.tolist() == expected
    assert (labels == 0).equal(x == 0)
