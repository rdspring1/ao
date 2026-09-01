"""DSV3 tensor-dump discovery and block-type classification.

Filename convention (the DSV3 dump script's actual
``f"{tensor_type}_rank{rank}_layer{layer_num}_{name}{suffix}_step{step}.pt"``,
``suffix`` being ``_expert{N}`` for routed experts): tensor type first, then
kitchen-style ``rank``/``layer``/``name``/``step`` fields, e.g.
``G_rank0_layer26_grouped_mlp_fc2_expert0_step0.pt``.

``name`` is one of a fixed, dump-side FQN-to-name map, not free text: attention
is ``qkv``/``kv_a``/``kv_b``/``proj``; dense FFN (layer 0 only) is
``fc1``/``fc2``/``fc3``; shared experts are ``shared_fc1``/``shared_fc2``/
``shared_fc3``; routed experts are ``grouped_mlp_fc1``/``grouped_mlp_fc2``/
``grouped_mlp_fc3`` plus an ``_expert{N}`` suffix. Anything outside this map is
never written by the dump script, so ``classify`` only needs to recognize
these four families — no mamba layers, no layer-index-driven lookup.
"""
from __future__ import annotations

import dataclasses
import os
import re
from typing import List, Optional, Sequence

import torch

_FILENAME_RE = re.compile(
    r"^(?P<type>X|W|G)_rank(?P<rank>\d+)_layer(?P<layer>\d+)_(?P<module>.+?)"
    r"(?:_expert(?P<expert>\d+))?_step(?P<step>\d+)\.pt$"
)


@dataclasses.dataclass(frozen=True)
class DsV3TensorInfo:
    tensor_type: str
    rank: int
    layer_num: int
    module_name: str
    expert_num: Optional[int]
    step: int
    filepath: str


def discover_dsv3_tensors(
    base_dir: str,
    tensor_type: str,
    *,
    rank: int = 0,
    step: int = 0,
    layer_names: Optional[Sequence[str]] = None,
    skip_layer_numbers: Optional[Sequence[int]] = None,
    exclude_experts: bool = False,
) -> List[DsV3TensorInfo]:
    """Discover DSV3 dump files matching ``tensor_type``/``rank``/``step``."""
    wanted_names = set(layer_names) if layer_names else None
    skip_layers = set(skip_layer_numbers) if skip_layer_numbers else set()
    infos: List[DsV3TensorInfo] = []
    for filename in os.listdir(base_dir):
        m = _FILENAME_RE.match(filename)
        if not m:
            continue
        if m.group("type") != tensor_type:
            continue
        if int(m.group("rank")) != rank or int(m.group("step")) != step:
            continue
        layer_num = int(m.group("layer"))
        if layer_num in skip_layers:
            continue
        module_name = m.group("module")
        if wanted_names is not None and module_name not in wanted_names:
            continue
        expert_num = m.group("expert")
        if expert_num is not None:
            if exclude_experts:
                continue
            expert_num = int(expert_num)
        infos.append(
            DsV3TensorInfo(
                tensor_type=tensor_type,
                rank=rank,
                layer_num=layer_num,
                module_name=module_name,
                expert_num=expert_num,
                step=step,
                filepath=os.path.join(base_dir, filename),
            )
        )
    infos.sort(key=lambda info: (info.layer_num, info.module_name, info.expert_num or -1))
    return infos


def load_dump_tensor(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected a bare tensor dump, got {type(tensor)}")
    return tensor.detach()


_ATTN_NAMES = frozenset({"qkv", "kv_a", "kv_b", "proj"})


def classify(module_name: str) -> str:
    """DSV3 block type for a dump's ``name`` field: attn, dense, moe shared/routed.

    ``name`` comes from the dump script's fixed FQN map (see module docstring),
    so this is a membership/prefix check against that map's four families, not
    a layer-index-driven lookup like kitchen's mamba/MoE/attn hybrid pattern.
    """
    if module_name in _ATTN_NAMES:
        return f"attn/{module_name}"
    if module_name.startswith("grouped_mlp_"):
        return f"moe/routed/{module_name[len('grouped_mlp_'):]}"
    if module_name.startswith("shared_"):
        return f"moe/shared/{module_name[len('shared_'):]}"
    return f"dense/{module_name}"
