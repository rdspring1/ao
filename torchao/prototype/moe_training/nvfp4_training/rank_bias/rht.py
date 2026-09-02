"""Random Hadamard Transform for the rotated lanes of kitchen recipes 9004 and 100483.

Transcribes ``QuantizeOpFP8HadamardTransform.perform_random_hadamard_transform_ref``
(``kitchen/quantization_with_hadamard_transform.py``) for the configurations
those two recipes use: ``enable_fused_hadamard_transform`` off on the reference
path, and ``perform_*_hadamard_transform_in_full_precision=False`` (so bf16 in,
bf16 out, fp32 accumulate).

Kitchen folds the ``1/sqrt(dim)`` into the cuBLAS ``alpha``, i.e. it scales the
fp32 accumulator before rounding to bf16, whereas :func:`transform` rounds and
then scales. The factor is a power of two at both dimensions used here, so the
two agree bit for bit outside the bf16 subnormal and overflow corners;
``test_kitchen_equivalence.py`` pins that against the real op rather than
trusting the argument.

Which lanes rotate is a property of the recipe, not of this module
(``get_rht_settings_for_tensor``):

* **9004** rotates the *transpose* lane only, at dim 16, from the ``wgrad``
  sign vector. Every identity lane is unrotated and W is never rotated.
* **100483** rotates *both* lanes of X and G at dim 128, because it enables the
  Hadamard transform on dgrad as well as wgrad. G's identity lane is the DGRAD
  GEMM and G's transpose lane is the WGRAD GEMM, so the two lanes use
  **different** sign vectors.

The two lanes also rotate along different axes. The transpose lane computes
``(x.T @ H).T``, mixing ``dim`` consecutive *rows* -- the same elements the
transpose-path QDQ blocks over. The identity lane computes ``x @ H``, mixing
``dim`` consecutive *columns*.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

HADAMARD_DIM = 16


# kitchen/ops/data/hadamard_random_sign_vec.json. Both recipes read the checked-in
# vectors; the entry is keyed by GEMM type and then by Hadamard dimension.
#
# 9004 leaves enable_online_randomization off, so ["wgrad"]["16"] is what every
# rank and every run uses, and it is the only sign vector that reproduces a 9004
# run bitwise. 100483 *does* enable online randomization, so its real runs
# resample per iteration; the checked-in ["128"] vectors are the reproducible
# stand-in, and ``--vary-rht-sign`` is the closer model of a training run.
WGRAD_SIGN: Tuple[int, ...] = (
    1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1,
)

WGRAD_SIGN_128: Tuple[int, ...] = (
    -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1,
    -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
    -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
    1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1,
    -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1,
    1, 1, 1, 1, 1,
)

DGRAD_SIGN_128: Tuple[int, ...] = (
    1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
    -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1,
    1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
    1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1,
    -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1,
    1, -1,
)

# (gemm lane, hadamard dim) -> checked-in sign vector. G's identity lane is the
# DGRAD GEMM and its transpose lane is the WGRAD GEMM
# (``get_gemm_types_for_tensor``), so the two lanes of one tensor draw different
# vectors under 100483.
SIGN_VECTORS = {
    ("wgrad", 16): WGRAD_SIGN,
    ("wgrad", 128): WGRAD_SIGN_128,
    ("dgrad", 128): DGRAD_SIGN_128,
}


def hadamard(dim: int, device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard, ``H[i, j] = (-1) ** popcount(i & j)`` -- scipy's matrix.

    kitchen builds it with ``scipy.linalg.hadamard(dim)`` and casts to bf16
    (``kitchen/ops/hadamard.py``); the entries are +-1, so the cast is exact.
    """
    index = torch.arange(dim, device=device)
    parity = (index[:, None] & index[None, :]).unsqueeze(0).bitwise_and(
        (1 << torch.arange(dim.bit_length(), device=device))[:, None, None]
    ).ne(0).sum(0)
    return torch.where(parity % 2 == 0, 1.0, -1.0).to(torch.bfloat16)


def sign_vector(
    device: torch.device,
    *,
    seed: Optional[int] = None,
    dim: int = HADAMARD_DIM,
    lane: str = "wgrad",
) -> torch.Tensor:
    """The +-1 diagonal. ``seed=None`` is kitchen's checked-in vector for the lane.

    An integer seed re-draws it the way kitchen's online randomization does
    (``kitchen/ops/hadamard.py`` ``random_sign_vec``: sign of a normal sample,
    with zeros pushed to +1), but from an explicit generator so a trial stays
    reproducible. Kitchen draws from the ambient generator, so a re-drawn sign
    vector matches kitchen's distribution, never its bits.

    ``lane`` is the GEMM the vector belongs to, so that the two lanes of one
    tensor stay independent -- and, when re-drawn, decorrelated.
    """
    if seed is None:
        return torch.tensor(SIGN_VECTORS[(lane, dim)], dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn(dim, device=device, generator=generator)
    return ((sample.sign() + 1e-7).sign()).to(torch.bfloat16)


def transform_matrices(sign: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(forward, inverse)`` = ``(diag(s) @ H, H @ diag(s))``.

    kitchen's ``fuse_hadamard_components`` / ``fuse_inverse_hadamard_components``.
    H is symmetric, so ``forward @ inverse * (1/dim) == I``.
    """
    matrix = hadamard(sign.numel(), sign.device)
    diagonal = torch.diagflat(sign)
    return diagonal @ matrix, matrix @ diagonal


def pad_rows(x: torch.Tensor, dim: int = HADAMARD_DIM) -> torch.Tensor:
    """Zero-pad rows up to a multiple of ``dim`` (kitchen's ``_pad_inputs_for_rht``)."""
    pad = -x.shape[0] % dim
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, pad), value=0.0).contiguous()


def pad_cols(x: torch.Tensor, dim: int = HADAMARD_DIM) -> torch.Tensor:
    """Zero-pad columns up to a multiple of ``dim``, for the identity lane."""
    pad = -x.shape[1] % dim
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad), value=0.0).contiguous()


def transform(
    x: torch.Tensor, matrix: torch.Tensor, *, transpose: bool = True
) -> torch.Tensor:
    """Rotate ``x`` by ``matrix`` along the lane's contraction axis.

    ``transpose=True`` is ``(x.T @ M).T``: groups of ``dim`` rows, the wgrad
    contraction axis, and the same axis the transpose-path QDQ blocks along, so
    an RHT tile and a quantization block cover the same elements.

    ``transpose=False`` is ``x @ M``: groups of ``dim`` columns, which is what
    the dgrad lane rotates and what the identity-path QDQ blocks along.

    kitchen passes ``1/sqrt(dim)`` as the cuBLAS ``alpha``, so the normalization
    is applied to the **fp32 accumulator, before** the rounding to bf16. That
    ordering is not cosmetic: at dim 16 the factor is 0.25 and scaling after the
    rounding gives the same bits, but at dim 128 it is ``1/(8*sqrt(2))``, which
    is not a power of two, and rounding first loses to it on roughly a third of
    the elements. ``test_rht_128_matches_kitchen`` is what pins this.
    """
    dim = matrix.shape[0]
    source = x.t().contiguous() if transpose else x.contiguous()
    accumulator = torch.matmul(source.view(-1, dim).float(), matrix.float())
    scaled = (accumulator * (1.0 / dim**0.5)).to(x.dtype)
    if transpose:
        return scaled.view(x.shape[::-1]).t().contiguous()
    return scaled.view(x.shape).contiguous()
