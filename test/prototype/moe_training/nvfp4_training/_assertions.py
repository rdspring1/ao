# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared assertions for the NVFP4 kernel tests.

Deliberately pytest-free so an out-of-tree comparison script can reuse them. The
vocabulary here is what each output can honestly be held to against the plain-PyTorch
reference in ``torchao.prototype.moe_training.nvfp4_training.nvfp4_reference``:

* **scales, amaxes -> bitwise.** Every step is IEEE-exact and torch does not enable fast
  math, so the reference reproduces the kernels' bytes exactly.
* **FP4 codes -> bracketed.** The kernels compute the per-block encode scale with
  ``rcp.approx.f32``, which torch cannot emit. ``assert_codes_bracketed`` quantizes at
  both ends of a few-ulp scale bracket and requires the kernel's code to be one of them;
  since the code is monotone in the scale, that admits exactly the codes an approximate
  reciprocal could produce and nothing else. Where the bracket is degenerate (the
  overwhelming majority of elements) it *is* bitwise equality.
"""

from typing import Optional

import torch

from torchao.prototype.moe_training.nvfp4_training.nvfp4_reference import (
    NVFP4ReferenceOutput,
    encode_scale_bracket,
    global_encode_scale,
    pack_fp4,
)

# E2M1 midpoints; a code can only flip if the scaled magnitude sits on one of these.
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def assert_scales_bitwise(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Compare E4M3 scales as raw bytes.

    Both sides are flattened: the kernels return 4-D swizzled ``(M//128, N//64, 32, 16)``
    while ``to_blocked`` returns a flat buffer, and the byte sequence is what matters.
    """
    got_b = got.flatten().contiguous().view(torch.uint8)
    ref_b = ref.flatten().contiguous().view(torch.uint8)
    assert got_b.shape == ref_b.shape, (
        f"{label}: shape mismatch {tuple(got_b.shape)} vs {tuple(ref_b.shape)}"
    )
    assert torch.equal(got_b, ref_b), (
        f"{label}: {(got_b != ref_b).sum().item()}/{got_b.numel()} fp8 scale bytes differ"
    )


def unpack_fp4_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """(R, C//2) packed uint8 -> (R, C) nibbles, undoing ``pack_uint4``'s even-in-low order.

    Returns ``long``, not ``uint8``: callers subtract two code tensors, and in uint8 a
    difference of -1 wraps to 255.
    """
    lo = (codes & 0xF).long()
    hi = (codes >> 4).long()
    return torch.stack((lo, hi), dim=-1).reshape(codes.shape[0], -1)


def unpack_fp4_magnitudes(codes: torch.Tensor) -> torch.Tensor:
    """FP4 magnitude codes (0-7), sign bit stripped."""
    return unpack_fp4_nibbles(codes) & 0x7


def assert_codes_bracketed(
    got: torch.Tensor,
    ref: NVFP4ReferenceOutput,
    global_amax: torch.Tensor,
    label: str,
    *,
    ulps: float = 4.0,
) -> None:
    """Every FP4 code must be reachable from a few-ulp perturbation of the encode scale.

    Also requires the sign nibbles to be exactly equal — the encode scale is positive, so
    no reciprocal difference can flip a sign.

    Compared per *nibble*, not per byte: a byte packs two elements, and one may round to
    the low end of the bracket while its neighbour rounds to the high end, in which case
    the byte equals neither endpoint even though both elements are in range.

    Deliberately no cap on *how many* codes differ from the exactly-rounded reference:
    that fraction is a property of the data, not of the kernel. Integer inputs under a
    power-of-two amax put scaled values exactly on E2M1 midpoints, where an approximate
    reciprocal is a coin flip and several percent of codes legitimately differ. The
    bracket plus ``assert_mismatches_are_midpoints`` bound the divergence structurally,
    which is what a cap would only approximate.
    """
    s_enc = global_encode_scale(global_amax)
    lo, hi = encode_scale_bracket(ref.block_scale, s_enc, ulps=ulps)

    def _expand(t):
        return t.repeat_interleave(ref.block_rows, dim=0).repeat_interleave(16, dim=1)

    got_n = unpack_fp4_nibbles(got)
    lo_n = unpack_fp4_nibbles(pack_fp4(ref.values * _expand(lo)))
    hi_n = unpack_fp4_nibbles(pack_fp4(ref.values * _expand(hi)))
    outside = ~((got_n == lo_n) | (got_n == hi_n))
    assert not outside.any(), (
        f"{label}: {int(outside.sum())}/{outside.numel()} FP4 codes fall outside a "
        f"{ulps}-ulp encode-scale bracket -- that is a recipe difference, not rounding"
    )

    sign_bad = ((got ^ ref.codes) & 0x88) != 0
    assert not sign_bad.any(), (
        f"{label}: {int(sign_bad.sum())} code bytes differ in a sign nibble; the encode "
        "scale is positive, so no reciprocal difference can do that"
    )


def assert_mismatches_are_midpoints(
    got: torch.Tensor, ref: NVFP4ReferenceOutput, label: str, *, rel: float = 1e-6
) -> None:
    """Diagnostic: every differing nibble must sit on an E2M1 rounding midpoint.

    Turns a future regression into a one-line explanation instead of a percentage. Run it
    on the RHT paths, where ``rcp.approx`` actually bites. Compares per nibble, not per
    byte -- only one of a byte's two elements may have moved.
    """
    diff = unpack_fp4_nibbles(got) != unpack_fp4_nibbles(ref.codes)
    if not diff.any():
        return
    mag = ref.scaled.double().abs()[diff]
    on_mid = torch.zeros_like(mag, dtype=torch.bool)
    for mid in _FP4_MIDPOINTS:
        on_mid |= (mag - mid).abs() <= rel * mid
    on_mid |= mag > 6.0  # saturation region
    assert on_mid.all(), (
        f"{label}: {int((~on_mid).sum())} differing elements are not near an E2M1 "
        "midpoint, so the difference is not reciprocal rounding"
    )


def dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    is_swizzled: bool = True,
) -> torch.Tensor:
    """Dequantize NVFP4 codes + E4M3 scales back to float32."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    return (
        NVFP4Tensor(
            codes.contiguous(),
            scales.contiguous(),
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=is_swizzled,
        )
        .dequantize()
        .float()
    )


def assert_scales_finite(scales: torch.Tensor, label: str = "scales") -> None:
    """No lower-bound check: TE emits a zero per-vector scale for a zero or underflowing
    block, so pinning small scales to a nonzero floor would contradict the ground truth."""
    assert torch.isfinite(scales.to(torch.float32)).all(), f"{label} must be finite"


def assert_scales_adjacent(
    got: torch.Tensor, ref: torch.Tensor, label: str, *, max_ulps: int = 1
) -> None:
    """fp8 scale bytes equal or within ``max_ulps`` representable steps.

    For comparisons against ``mx_formats.nvfp4_quantize``, which multiplies by a
    reciprocal and applies an E4M3_EPS floor where the kernels follow TE's div_rn with no
    floor. Positive e4m3 bytes are magnitude-monotonic, so a byte delta is a ULP delta.
    """
    got_b = got.flatten().contiguous().view(torch.uint8).to(torch.int16)
    ref_b = ref.flatten().contiguous().view(torch.uint8).to(torch.int16)
    assert got_b.shape == ref_b.shape, (
        f"{label}: shape mismatch {tuple(got_b.shape)} vs {tuple(ref_b.shape)}"
    )
    diff = (got_b - ref_b).abs()
    assert (diff <= max_ulps).all(), (
        f"{label}: {(diff > max_ulps).sum().item()}/{diff.numel()} fp8 scale bytes "
        f"differ by >{max_ulps} ULP (max {diff.max().item()})"
    )


def assert_zero_quantized(
    codes: torch.Tensor, scales: torch.Tensor, dequantized: Optional[torch.Tensor] = None
) -> None:
    """An all-zero input packs to zero codes, stores a zero block scale, and dequantizes
    to exactly zero."""
    assert torch.count_nonzero(codes) == 0, "zero input must pack to zero codes"
    assert torch.count_nonzero(scales.to(torch.float32)) == 0, (
        "zero input must store a zero block scale"
    )
    if dequantized is not None:
        assert torch.count_nonzero(dequantized) == 0, "zero input must dequantize to zero"
