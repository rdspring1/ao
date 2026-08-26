# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared skip marks for the V2 / V1_REQUANT kernel tests.

Every one of these test modules needs the same two gates, so they live here rather
than being copied seven times:

* ``requires_sm100`` -- Triton, SM100, and PyTorch 2.10+, as elsewhere in this
  directory.
* ``kernel_gate`` -- the per-file switch that turns a kernel's numerics tests on once
  its ``@triton.jit`` body lands. The wrapper-layer tests (validation and
  ``register_fake``) deliberately do **not** go behind this gate: that layer is
  complete today and should be failing loudly if it regresses.
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.utils import is_sm_at_least_100, torch_version_at_least

TRITON_AVAILABLE = (
    has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0")
)

requires_sm100 = [
    pytest.mark.skipif(not has_triton(), reason="unsupported without triton"),
    pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+"),
    pytest.mark.skipif(
        not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
    ),
]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def maybe_sm100(fn):
    """Apply the hardware/version gates. Use for tests that need no kernel body."""
    for mark in requires_sm100:
        fn = mark(fn)
    return fn


def kernel_gate(implemented: bool, module: str):
    """Return a decorator gating a test on ``module``'s kernel body being written.

    Args:
        implemented: the module-level ``_KERNEL_IMPLEMENTED`` flag.
        module: the file whose ``@triton.jit`` body is still a stub, for the reason
            string.
    """
    skip = pytest.mark.skipif(
        not implemented, reason=f"Triton kernel body in {module} is still a stub"
    )

    def decorate(fn):
        return maybe_sm100(skip(fn))

    return decorate
