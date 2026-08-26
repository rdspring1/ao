# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic resampling of the V2 recipe's RHT sign vectors.

V1 and V1_REQUANT draw one 16-element sign vector at setup and never change it, so
the whole RHT matrix can be memoized by value. V2 cannot: it resamples ``dgrad_rht``
every optimizer step and ``wgrad_rht`` every accumulation microbatch, and a cache
keyed on a changing sign vector grows one entry per resample for the run's lifetime.

Two properties this module exists to guarantee:

* **Fixed shape, updated in place.** ``resample_nvfp4_rht_signs`` only ever calls
  ``buffer.copy_(...)``. The buffers keep their addresses, so a CUDA graph captured
  around the training step stays valid across resamples.
* **Derived, not communicated.** Every vector is a pure function of
  ``(seed, step, microbatch, module FQN, buffer name)``. Ranks agree by construction
  with no collective, and a run replays bitwise from its seed.

Which buffers are dynamic is read off their length, with no registration protocol:

======  ==========  =========================================
Length  Recipe      Cadence
======  ==========  =========================================
16      V1, V1_REQUANT  static -- never resampled
128     V2              ``_dgrad_*`` per step, others per microbatch
======  ==========  =========================================
"""

import hashlib

import torch
from torch import nn

# Buffer-name suffixes this module manages. A module opts in simply by registering a
# 128-element buffer under one of these names; there is no separate registry to keep
# in sync with the modules.
WGRAD_SIGN_BUFFER_SUFFIX = "_rht_sign_vector"
DGRAD_SIGN_BUFFER_SUFFIX = "_dgrad_rht_sign_vector"

# Only 128-element vectors are resampled. A 16-element vector belongs to a static
# recipe and must not be touched: V1 resolves it through the sign-keyed lru_cache,
# so mutating it would leak one cached RHT matrix per step.
_DYNAMIC_SIGN_LENGTH = 128


def _derive_seed(seed: int, step: int, microbatch: int, fqn: str, name: str) -> int:
    """Stable 63-bit seed for one (buffer, iteration) pair.

    BLAKE2b rather than ``hash()``: Python's string hash is salted per process, so it
    would give different sign vectors on different ranks for the same FQN.
    """
    key = f"{seed}|{step}|{microbatch}|{fqn}|{name}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def draw_sign_vector(length: int, seed: int) -> torch.Tensor:
    """Deterministic {-1, +1} int8 sign vector of the given length, on CPU.

    Drawn on CPU and copied to the device by the caller. Two reasons: the draw does
    not perturb the CUDA RNG stream that stochastic rounding reads from, and it stays
    reproducible regardless of how many devices participate. At 128 elements per
    buffer per microbatch the host-side cost is irrelevant.
    """
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8)
    return bits * 2 - 1


def iter_dynamic_sign_buffers(root: nn.Module):
    """Yield ``(fqn, name, buffer, is_dgrad)`` for every dynamic sign buffer under root."""
    for fqn, module in root.named_modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is None or buffer.numel() != _DYNAMIC_SIGN_LENGTH:
                continue
            is_dgrad = name.endswith(DGRAD_SIGN_BUFFER_SUFFIX)
            if not (is_dgrad or name.endswith(WGRAD_SIGN_BUFFER_SUFFIX)):
                continue
            yield fqn, name, buffer, is_dgrad


def resample_nvfp4_rht_signs(
    root: nn.Module,
    *,
    seed: int,
    step: int,
    microbatch: int = 0,
) -> int:
    """Resample every dynamic RHT sign buffer under ``root``, in place.

    ``dgrad_rht`` buffers advance with ``step`` only, so they are constant across the
    microbatches of one optimizer step; ``wgrad_rht`` buffers advance with both. That
    asymmetry is the recipe's, not an implementation detail: the wgrad GEMM's two
    operands are produced in the same microbatch and cancel within it, while the
    dgrad GEMM's weight operand is requantized once per step.

    Call this *outside* any CUDA-graph capture, before the forward pass of each
    microbatch. Returns the number of buffers updated, which is 0 for a model with no
    V2 layers -- callers can use that to detect a mis-wired training loop.

    Note the current limitation: this must be driven by the training loop, which has
    to know both the optimizer step and the microbatch index. A caller that only
    advances ``microbatch`` and leaves ``step`` at 0 gets correct-but-more-frequent
    dgrad resampling, not incorrect gradients.
    """
    updated = 0
    for fqn, name, buffer, is_dgrad in iter_dynamic_sign_buffers(root):
        # dgrad is pinned to the step so it is stable across the microbatches that
        # accumulate into one update.
        effective_microbatch = 0 if is_dgrad else microbatch
        derived = _derive_seed(seed, step, effective_microbatch, fqn, name)
        signs = draw_sign_vector(_DYNAMIC_SIGN_LENGTH, derived)
        # copy_, never assignment: the buffer address must survive graph capture.
        buffer.copy_(signs.to(device=buffer.device, dtype=buffer.dtype))
        updated += 1
    return updated
