# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Single-GPU NVFP4 training example for one DeepSeek-V3 671B expert shard.

The input is pre-routed: its rows must be contiguous by expert in the same
order as ``num_tokens_per_expert``. This example intentionally has no router,
permutation, distributed initialization, or expert-parallel collectives.

Run with::

    python -m torchao.prototype.moe_training.nvfp4_training.nvfp4_single_gpu_example
    python -m ...nvfp4_single_gpu_example --recipe v2
    python -m ...nvfp4_single_gpu_example --recipe moe_split

``--recipe moe_split`` is design doc §17's routing: FC1 (``w1``/``w3``) on
V1_REQUANT, FC2 (``w2``) on V2, with independent sign vectors and seeds per layer.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

NUM_LOCAL_EXPERTS = 128
MODEL_DIM = 7168
EXPERT_HIDDEN_DIM = 2048
TOKENS_PER_EXPERT = 128
NUM_PACKED_ROWS = NUM_LOCAL_EXPERTS * TOKENS_PER_EXPERT
RHT_SIGN_VECTOR = tuple(1 if index % 2 == 0 else -1 for index in range(16))
RECIPES = ("v1", "v1_requant", "v2", "moe_split")


class SimplifiedMoE(nn.Module):
    """MoE layer whose input has already been routed and packed by expert."""

    def __init__(self, device: torch.device, recipe: str = "v1"):
        super().__init__()
        self.experts = GroupedExperts(device, recipe)

    def forward(
        self, routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        return self.experts(routed_input, num_tokens_per_expert)


class GroupedExperts(nn.Module):
    """DeepSeek-V3 experts backed by differentiable NVFP4 grouped GEMMs."""

    def __init__(self, device: torch.device, recipe: str = "v1"):
        super().__init__()
        if recipe not in RECIPES:
            raise ValueError(f"recipe must be one of {RECIPES}, got {recipe!r}")
        self.recipe = recipe
        parameter_kwargs = {"device": device, "dtype": torch.bfloat16}
        self.w1 = nn.Parameter(
            torch.empty(
                NUM_LOCAL_EXPERTS,
                EXPERT_HIDDEN_DIM,
                MODEL_DIM,
                **parameter_kwargs,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                NUM_LOCAL_EXPERTS,
                MODEL_DIM,
                EXPERT_HIDDEN_DIM,
                **parameter_kwargs,
            )
        )
        self.w3 = nn.Parameter(
            torch.empty(
                NUM_LOCAL_EXPERTS,
                EXPERT_HIDDEN_DIM,
                MODEL_DIM,
                **parameter_kwargs,
            )
        )
        # FC1 and FC2 get independent seeds and sign vectors. Under the split recipe
        # they no longer share a quantization path, so sharing either would correlate
        # their noise; §17 makes independence a requirement rather than a preference.
        self.register_buffer(
            "sr_seed", torch.tensor([1234], dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "fc2_sr_seed", torch.tensor([5678], dtype=torch.int64, device=device)
        )
        # 128-element buffers are resampled in place by resample_nvfp4_rht_signs;
        # a fresh allocation per step would break CUDA-graph capture.
        for name, seed in (("fc2_wgrad", 1), ("fc2_dgrad", 2)):
            generator = torch.Generator().manual_seed(seed)
            bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
            self.register_buffer(f"_{name}_rht_sign_vector", (bits * 2 - 1).to(device))

        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.normal_(self.w3, mean=0.0, std=0.02)

    def _grouped_mm(self, x, weight, recipe, *, seed):
        """Dispatch one grouped GEMM to the recipe's entrypoint."""
        if recipe == "v1":
            from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
                _to_nvfp4_rht_rs_then_scaled_grouped_mm,
            )

            return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                x,
                weight,
                RHT_SIGN_VECTOR,
                seed,
                offs=self._offsets,
                pad_token_groups_for_grouped_mm=False,
            )

        from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
            nvfp4_v1_requant_grouped_mm,
            nvfp4_v2_grouped_mm,
        )

        if recipe == "v2":
            return nvfp4_v2_grouped_mm(
                x,
                weight,
                wgrad_rht=self._fc2_wgrad_rht_sign_vector,
                dgrad_rht=self._fc2_dgrad_rht_sign_vector,
                sr_seed=seed,
                offs=self._offsets,
                pad_token_groups_for_grouped_mm=False,
            )
        return nvfp4_v1_requant_grouped_mm(
            x,
            weight,
            sign_vector=RHT_SIGN_VECTOR,
            sr_seed=seed,
            offs=self._offsets,
            pad_token_groups_for_grouped_mm=False,
        )

    def forward(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        self._offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        # §17: FC1 on V1_REQUANT, FC2 on V2. Any other value runs one recipe
        # throughout, which is what makes the split a configuration choice.
        if self.recipe == "moe_split":
            fc1_recipe, fc2_recipe = "v1_requant", "v2"
        else:
            fc1_recipe = fc2_recipe = self.recipe

        gate = self._grouped_mm(x, self.w1, fc1_recipe, seed=self.sr_seed)
        up = self._grouped_mm(x, self.w3, fc1_recipe, seed=self.sr_seed)
        hidden = F.silu(gate) * up
        return self._grouped_mm(hidden, self.w2, fc2_recipe, seed=self.fc2_sr_seed)


def main(recipe: str = "v1") -> None:
    if not torch.cuda.is_available():
        print("Skipping NVFP4 example: CUDA is not available.")
        return
    if not has_triton():
        print("Skipping NVFP4 example: Triton is not available.")
        return
    if not torch_version_at_least("2.13.0"):
        print("Skipping NVFP4 example: PyTorch 2.13 or newer is required.")
        return

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability < (10, 0):
        print("Skipping NVFP4 example: an SM100 or newer GPU is required.")
        return

    torch.manual_seed(42)
    model = SimplifiedMoE(device, recipe)
    routed_input = torch.randn(
        NUM_PACKED_ROWS,
        MODEL_DIM,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    num_tokens_per_expert = torch.full(
        (NUM_LOCAL_EXPERTS,),
        TOKENS_PER_EXPERT,
        device=device,
        dtype=torch.int32,
    )

    output = model(routed_input, num_tokens_per_expert)
    loss = output.float().square().mean()
    loss.backward()

    expected_output_shape = (NUM_PACKED_ROWS, MODEL_DIM)
    assert output.shape == expected_output_shape
    assert routed_input.grad is not None and torch.isfinite(routed_input.grad).all()
    print(f"output: {tuple(output.shape)}")
    print(f"input grad: {tuple(routed_input.grad.shape)} (finite)")
    for name, parameter in model.experts.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        print(f"{name} grad: {tuple(parameter.grad.shape)} (finite)")
    print(
        f"NVFP4 single-GPU forward/backward completed successfully (recipe={recipe})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="v1", choices=RECIPES)
    main(parser.parse_args().recipe)
