"""Random Hadamard Transform for the wgrad lane of kitchen recipe 9004 (== 6304).

Transcribes ``QuantizeOpFP8HadamardTransform.perform_random_hadamard_transform_ref``
(``kitchen/quantization_with_hadamard_transform.py``) for the configuration 9004
uses: ``hadamard_dim_wgrad=16``, ``enable_fused_hadamard_transform=False`` (so the
cuBLAS reference path, not the fused kernel) and
``perform_wgrad_hadamard_transform_in_full_precision=False`` (so bf16 in, bf16
out, fp32 accumulate).

Kitchen folds the ``1/sqrt(16)`` into the cuBLAS ``alpha``, i.e. it scales the
fp32 accumulator before rounding to bf16, whereas :func:`transform` rounds and
then scales. 0.25 is a power of two, so the two agree bit for bit outside the
bf16 subnormal and overflow corners; ``test_kitchen_equivalence.py`` pins that
against the real op rather than trusting the argument.

Only the transpose lane is modelled: for 9004 the identity lane of every tensor
is unrotated (``perform_hadamard_transform_fprop`` and ``_dgrad`` are both
False), and W is never rotated at all.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

HADAMARD_DIM = 16

# kitchen/ops/data/hadamard_random_sign_vec.json, ["wgrad"]["16"]. Recipe 9004
# leaves enable_online_randomization off, so this vector is what every rank and
# every run uses; it is the only sign vector that reproduces a 9004 run bitwise.
WGRAD_SIGN: Tuple[int, ...] = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)


def hadamard_16(device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard, ``H[i, j] = (-1) ** popcount(i & j)`` -- scipy's matrix.

    kitchen builds it with ``scipy.linalg.hadamard(16)`` and casts to bf16
    (``kitchen/ops/hadamard.py``); the entries are +-1, so the cast is exact.
    """
    index = torch.arange(HADAMARD_DIM, device=device)
    parity = (index[:, None] & index[None, :]).bitwise_and(
        torch.tensor([1, 2, 4, 8], device=device)[:, None, None]
    ).ne(0).sum(0)
    return torch.where(parity % 2 == 0, 1.0, -1.0).to(torch.bfloat16)


def sign_vector(device: torch.device, *, seed: Optional[int] = None) -> torch.Tensor:
    """The +-1 diagonal. ``seed=None`` is kitchen's static wgrad vector.

    An integer seed re-draws it the way kitchen's online randomization does
    (``kitchen/ops/hadamard.py`` ``random_sign_vec``: sign of a normal sample,
    with zeros pushed to +1), but from an explicit generator so a trial stays
    reproducible. Kitchen draws from the ambient generator, so a re-drawn sign
    vector matches kitchen's distribution, never its bits.
    """
    if seed is None:
        return torch.tensor(WGRAD_SIGN, dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn(HADAMARD_DIM, device=device, generator=generator)
    return ((sample.sign() + 1e-7).sign()).to(torch.bfloat16)


def transform_matrices(sign: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(forward, inverse)`` = ``(diag(s) @ H, H @ diag(s))``.

    kitchen's ``fuse_hadamard_components`` / ``fuse_inverse_hadamard_components``.
    H is symmetric at dim 16, so ``forward @ inverse * (1/16) == I``.
    """
    hadamard = hadamard_16(sign.device)
    diagonal = torch.diagflat(sign)
    return diagonal @ hadamard, hadamard @ diagonal


def pad_rows(x: torch.Tensor) -> torch.Tensor:
    """Zero-pad rows up to a multiple of 16 (kitchen's ``_pad_inputs_for_rht``)."""
    pad = -x.shape[0] % HADAMARD_DIM
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, pad), value=0.0).contiguous()


def transform(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """``(x.T @ M).T``: rotate groups of 16 rows, i.e. the wgrad contraction axis.

    Same axis the transpose-path QDQ blocks along, so an RHT tile and a
    quantization block cover the same 16 elements.
    """
    rotated = torch.matmul(x.t().contiguous().view(-1, HADAMARD_DIM), matrix)
    scaled = rotated * (1.0 / HADAMARD_DIM**0.5)
    return scaled.view(x.shape[::-1]).t().contiguous()
