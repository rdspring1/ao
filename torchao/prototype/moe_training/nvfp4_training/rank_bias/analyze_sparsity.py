#!/usr/bin/env python3
"""How much of a tensor NVFP4 throws away: exact zeros, and elements that flush to FP4 0.

Companion to :mod:`analyze_rank_bias`, deliberately a separate script. That one
is a *trials* sweep -- it measures how the reconstruction error of a stochastic
recipe decays as trials are averaged, so its whole shape is the T loop, the
per-bucket MSE curves and the RNG. Sparsity is a *static* property of the
tensor: no trials, no averaging, and (for the raw numbers) no recipe at all.
The two share the leaf quantizers, ``dsv3_dumps`` discovery and the recipe
table; they share no control flow.

Three numbers, which are not the same thing and are routinely conflated:

``exact_zero``
    Elements that are exactly 0 in the dump. A property of the *model*: ReLU
    families produce them, SiLU/GELU families essentially never do, and MoE
    routing produces them in whole rows.

``flush``
    Elements that are nonzero in the dump but quantize to FP4 code 0. This is
    what NVFP4 actually discards, and unlike ``exact_zero`` it is *block
    relative*: an element flushes when it is small compared to the amax of its
    own 16-element block, not when it is small in absolute terms. An element at
    1e-30 in a block whose amax is 1e-30 survives; an element at 1.0 in a block
    whose amax is 100 does not.

``dead_block``
    Blocks whose E4M3 block scale itself underflows to zero, so the entire block
    reconstructs as zero regardless of its contents.

The distributional statistic behind ``flush`` is ``|x| / block_amax``. An
element rounds to FP4 zero at roughly ``|x| < block_amax / 24`` (the E2M1 RNE
threshold 0.25 against the 6.0 grid ceiling), so ``p50_rel`` -- the median of
that ratio -- is the single number that says how much headroom a typical
element has. It is recipe-independent, which makes it the right thing to compare
across models.

**The RHT interacts with all of this.** A Hadamard rotation mixes 16 (9004) or
128 (100483) elements per output, so it is precisely a sparsity-*destroying*
transform: a rotated lane has far fewer exact zeros and a much tighter
``|x|/block_amax`` distribution than the same tensor unrotated. Sparsity is
therefore reported per lane, with the rotation the recipe actually applies to
that lane, and the raw (pre-rotation) exact-zero fraction is reported alongside
so the model property stays visible.

Example:
    # every G dump in a tree, both lanes, under recipe 9004
    python analyze_sparsity.py --base-dir ./dumps --tensor-type G \\
        --variants G G.T --csv sparsity.csv

    # no dumps handy: synthesize fc1/fc2/fc3 gradients for a GLU vs a
    # non-GLU MLP and compare, which is what isolates the activation's effect
    python analyze_sparsity.py --synthetic --activation swiglu relu2 \\
        --csv synthetic_sparsity.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from . import rht
from .analyze_rank_bias import (
    BACKENDS,
    GEMM_TYPES,
    RECIPE_ALIASES,
    RECIPES,
    VARIANTS,
    flatten_to_2d,
    rht_matrices,
)
from .nvfp4_reference import BLOCK_SIZE

# |x| / block_amax below which an E2M1 RNE cast returns zero: the grid ceiling is
# 6.0 and the RNE threshold to zero is 0.25, so 0.25 / 6.0 = 1/24. Exact only in
# the limit of an unrounded block scale; ``flush`` measures the real thing.
FP4_ZERO_RATIO = 0.25 / 6.0


def _sr_kwarg(backend) -> str:
    """Which keyword this quantizer spells stochastic rounding with.

    ``nvfp4_*`` randomizes the FP4 *data* and calls it ``use_sr``; ``eden_*``
    randomizes the E4M3 *block scale* and calls it ``stochastic_round_scale``.
    The names differ because the things differ, so this adapts rather than
    renaming one of them into the other.
    """
    if "stochastic_round_scale" in inspect.signature(backend.quantize).parameters:
        return "stochastic_round_scale"
    return "use_sr"


def _logical(x: torch.Tensor, transpose: bool) -> torch.Tensor:
    """View whose last dim is the 16-element quantization-block axis."""
    return x.t() if transpose else x


def _percentile(values: torch.Tensor, q: float) -> float:
    """``q``-th percentile of a 1-D fp32 tensor, computed on GPU."""
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values, q))


@dataclasses.dataclass
class SparsityStats:
    numel: int
    raw_exact_zero: float      # of the tensor as dumped, before any rotation
    exact_zero: float          # of the tensor as fed to the quantizer
    flush: float               # nonzero in, FP4 code 0 out
    fp4_zero: float            # code 0 out, whatever went in
    dead_block: float
    p50_rel: float             # median |x| / block_amax
    p05_rel: float
    below_threshold: float     # analytic |x| / block_amax < 1/24

    def row(self) -> Dict[str, object]:
        return {
            "numel": self.numel,
            "raw_exact_zero_pct": 100.0 * self.raw_exact_zero,
            "exact_zero_pct": 100.0 * self.exact_zero,
            "flush_pct": 100.0 * self.flush,
            "fp4_zero_pct": 100.0 * self.fp4_zero,
            "dead_block_pct": 100.0 * self.dead_block,
            "p50_rel": self.p50_rel,
            "p05_rel": self.p05_rel,
            "below_1_24_pct": 100.0 * self.below_threshold,
        }


def sparsity_stats(
    tensor: torch.Tensor,
    *,
    transpose: bool,
    backend,
    matrices: Optional[Tuple[torch.Tensor, torch.Tensor]],
    rht_dim: int,
    use_sr: bool = False,
    seed: int = 0,
) -> SparsityStats:
    """Sparsity of ``tensor`` as the recipe's quantizer actually sees it.

    ``matrices`` is the rotation for this lane, or None when the recipe leaves
    it unrotated; when given, every statistic except ``raw_exact_zero`` is
    measured *after* the rotation, because that is what gets quantized.

    Stochastic rounding is off by default. It moves elements across the FP4 zero
    boundary at random, so a flush fraction measured under SR is a sample rather
    than a property; the RNE number is the one that is reproducible and the one
    worth comparing across tensors.
    """
    # Integer count over an fp32 .mean(): exact, and directly comparable with
    # ``exact_zero`` below, which is also an integer ratio. A float mean over a
    # multi-million-element tensor loses low bits to the reduction.
    raw_exact_zero = int((tensor == 0).sum()) / tensor.numel()

    if matrices is None:
        q_input = tensor
        valid_rows, valid_cols = tensor.shape
    else:
        forward, _ = matrices
        padded = (
            rht.pad_rows(tensor, rht_dim)
            if transpose
            else rht.pad_cols(tensor, rht_dim)
        )
        q_input = rht.transform(padded, forward, transpose=transpose)
        valid_rows, valid_cols = tensor.shape

    quantized = backend.quantize(
        q_input, transpose=transpose, seed=seed, **{_sr_kwarg(backend): use_sr}
    )
    codes = quantized.data_q
    block_descale = quantized.block_descale

    # Everything below lives in the logical (block-along-last-dim) layout, and
    # the quantizer may have padded, so build a validity mask in that layout
    # rather than trusting the shapes to line up.
    view = _logical(q_input, transpose)
    n_rows, n_cols = codes.shape
    mask = torch.zeros((n_rows, n_cols), dtype=torch.bool, device=codes.device)
    logical_valid = (valid_cols, valid_rows) if transpose else (valid_rows, valid_cols)
    mask[: logical_valid[0], : logical_valid[1]] = True

    padded_view = torch.zeros((n_rows, n_cols), dtype=torch.float32, device=codes.device)
    padded_view[: view.shape[0], : view.shape[1]] = view.float()

    total = int(mask.sum())
    is_zero_in = (padded_view == 0) & mask
    is_zero_out = (codes == 0) & mask

    blocked = padded_view.reshape(n_rows, n_cols // BLOCK_SIZE, BLOCK_SIZE)
    block_mask = mask.reshape(n_rows, n_cols // BLOCK_SIZE, BLOCK_SIZE)
    block_amax = blocked.abs().amax(dim=-1, keepdim=True)
    # 0/0 -> 0: an all-zero block has no headroom to measure.
    rel = torch.where(block_amax > 0, blocked.abs() / block_amax, torch.zeros_like(blocked))
    rel_valid = rel[block_mask & (blocked != 0)]

    live_blocks = (block_amax.squeeze(-1) > 0) & block_mask.any(dim=-1)
    n_live = int(live_blocks.sum())
    dead = (
        float(((block_descale == 0) & live_blocks).sum()) / n_live if n_live else 0.0
    )

    n_zero_in = int(is_zero_in.sum())
    return SparsityStats(
        numel=total,
        raw_exact_zero=raw_exact_zero,
        exact_zero=n_zero_in / total if total else 0.0,
        flush=float((is_zero_out & ~is_zero_in).sum()) / total if total else 0.0,
        fp4_zero=float(is_zero_out.sum()) / total if total else 0.0,
        dead_block=dead,
        p50_rel=_percentile(rel_valid, 0.50),
        p05_rel=_percentile(rel_valid, 0.05),
        below_threshold=(
            float((rel_valid < FP4_ZERO_RATIO).sum()) / total if total else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic fc1 / fc2 / fc3 gradients
# ---------------------------------------------------------------------------

# Each entry maps an activation to (has_fc3, gradient builder). The builder gets
# the pre-activations and the incoming gradient and returns the gradient at each
# linear's *output*, which is the tensor kitchen quantizes as that layer's G.
ACTIVATIONS = ("swiglu", "geglu", "relu2", "gelu", "silu")


def synthetic_mlp_grads(
    activation: str,
    *,
    tokens: int,
    hidden: int,
    ffn: int,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Gradients at fc1 / fc2 / fc3 outputs for one MLP block.

    Gated (``swiglu`` / ``geglu``) builds ``fc2(act(fc1(x)) * fc3(x))``; the rest
    build ``fc2(act(fc1(x)))`` and have no fc3. The point of this generator is
    that it isolates the activation: the shapes, the input distribution and the
    weight init are identical across choices, so any difference in the sparsity
    table is the nonlinearity and nothing else.

    G for a linear is the gradient with respect to its **output**, which is what
    the dgrad and wgrad GEMMs consume:

    * ``G_fc2 = dy`` -- never touched by the activation, so it is the control.
    * ``G_fc1 = da * d(act)/d(h1)`` (gated: times ``h3``)
    * ``G_fc3 = da * act(h1)`` -- gated only.

    ``d(act)/d(h1)`` is where the exact zeros come from: ``relu2`` differentiates
    to ``2*relu(h1)``, which is *exactly* zero on half its domain, while SiLU and
    GELU derivatives are never exactly zero.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *s: torch.randn(*s, generator=g, device=device)

    x = randn(tokens, hidden)
    w1 = randn(ffn, hidden) / hidden**0.5
    w2 = randn(hidden, ffn) / ffn**0.5
    h1 = x @ w1.t()

    gated = activation in ("swiglu", "geglu")
    if gated:
        w3 = randn(ffn, hidden) / hidden**0.5
        h3 = x @ w3.t()

    if activation in ("swiglu", "silu"):
        sig = torch.sigmoid(h1)
        act = h1 * sig
        dact = sig * (1 + h1 * (1 - sig))
    elif activation in ("geglu", "gelu"):
        # tanh-free exact GELU, matching torch's default 'none' approximation.
        cdf = 0.5 * (1 + torch.erf(h1 / 2**0.5))
        pdf = torch.exp(-0.5 * h1 * h1) / (2 * torch.pi) ** 0.5
        act = h1 * cdf
        dact = cdf + h1 * pdf
    elif activation == "relu2":
        relu = torch.relu(h1)
        act = relu * relu
        dact = 2 * relu
    else:
        raise ValueError(f"unknown activation {activation!r}")

    a = act * h3 if gated else act
    dy = randn(tokens, hidden)
    da = dy @ w2  # gradient at the elementwise-activation output

    grads = {"fc2": dy, "fc1": da * dact * h3 if gated else da * dact}
    if gated:
        grads["fc3"] = da * act
    return {k: v.to(torch.bfloat16) for k, v in grads.items()}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FIELDS = [
    "tensor",
    "family",
    "module",
    "recipe_id",
    "variant",
    "rht",
    "numel",
    "raw_exact_zero_pct",
    "exact_zero_pct",
    "flush_pct",
    "fp4_zero_pct",
    "dead_block_pct",
    "p50_rel",
    "p05_rel",
    "below_1_24_pct",
]


def analyze_one(
    tensor: torch.Tensor,
    *,
    name: str,
    family: str,
    module: str,
    recipe_id: str,
    variant: str,
    tensor_type: str,
    backends: Dict[str, object],
    use_sr: bool,
    seed: int,
) -> Dict[str, object]:
    recipe = RECIPES[recipe_id]
    _, transpose = VARIANTS[variant]
    gemm = GEMM_TYPES[(tensor_type, transpose)]
    use_rht = gemm in recipe.rht_gemms
    matrices = (
        rht_matrices(tensor.device, None, recipe.rht_dim, gemm) if use_rht else None
    )
    stats = sparsity_stats(
        tensor,
        transpose=transpose,
        backend=backends[recipe.quantizer[tensor_type]],
        matrices=matrices,
        rht_dim=recipe.rht_dim,
        use_sr=use_sr,
        seed=seed,
    )
    row = {
        "tensor": name,
        "family": family,
        "module": module,
        "recipe_id": recipe_id,
        "variant": variant,
        "rht": f"{recipe.rht_dim}/{gemm}" if use_rht else "none",
    }
    row.update(stats.row())
    return row


def print_table(rows: List[Dict[str, object]], group_by: str) -> None:
    """Median of each statistic within a group, widest-flush first."""
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[(str(r[group_by]), str(r["variant"]))].append(r)

    header = (
        f"{'group':>22} {'variant':>7} {'rht':>10} {'n':>4} "
        f"{'raw0%':>8} {'flush%':>8} {'fp4_0%':>8} {'dead%':>7} "
        f"{'p50_rel':>8} {'p05_rel':>8}"
    )
    print(header)
    print("-" * len(header))
    med = lambda key, rs: statistics.median([float(r[key]) for r in rs])
    for (group, variant), rs in sorted(
        groups.items(), key=lambda kv: -med("flush_pct", kv[1])
    ):
        print(
            f"{group:>22} {variant:>7} {str(rs[0]['rht']):>10} {len(rs):>4} "
            f"{med('raw_exact_zero_pct', rs):8.3f} {med('flush_pct', rs):8.3f} "
            f"{med('fp4_zero_pct', rs):8.3f} {med('dead_block_pct', rs):7.3f} "
            f"{med('p50_rel', rs):8.4f} {med('p05_rel', rs):8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact-zero and FP4 flush-to-zero sparsity of NVFP4 operands."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-dir", help="Tree of DSV3 .pt dumps to discover.")
    source.add_argument("--tensor-path", help="A single bare-tensor .pt dump.")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Synthesize fc1/fc2/fc3 gradients instead of reading dumps.",
    )
    parser.add_argument(
        "--activation",
        nargs="+",
        default=["swiglu", "relu2"],
        choices=ACTIVATIONS,
        help="Synthetic MLP activations to compare (default: swiglu relu2, i.e. "
        "a GLU against a non-GLU).",
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn", type=int, default=8192)
    parser.add_argument("--tensor-type", default="G", choices=["X", "W", "G"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["G", "G.T"],
        choices=list(VARIANTS),
        help="Lanes to report; each gets the rotation its recipe applies to it.",
    )
    parser.add_argument(
        "--recipe",
        default="9004",
        choices=sorted(set(RECIPES) | set(RECIPE_ALIASES)),
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="cutedsl")
    parser.add_argument(
        "--use-sr",
        action="store_true",
        help="Quantize with stochastic rounding. Off by default: SR moves "
        "elements across the FP4 zero boundary at random, so the flush fraction "
        "becomes a sample rather than a property of the tensor.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    # Same discovery filters plot_bias_heatmaps.py exposes; a full DSV3 tree is
    # 345 G tensors, so subsetting it is the normal case, not an edge case.
    parser.add_argument("--layer-types", nargs="+", default=None)
    parser.add_argument("--skip-layer-numbers", type=int, nargs="+", default=None)
    parser.add_argument("--exclude-experts", action="store_true")
    parser.add_argument("--csv", default=None, help="Write the per-tensor rows here.")
    parser.add_argument(
        "--group-by",
        default="family",
        choices=["family", "module"],
        help="Aggregation key for the printed table. 'family' is the DSV3 block "
        "class for dumps and activation/layer for synthetic runs; 'module' "
        "collapses across both (default: family).",
    )
    args = parser.parse_args()

    recipe_id = RECIPE_ALIASES.get(args.recipe, args.recipe)
    for variant in args.variants:
        reason = RECIPES[recipe_id].unsupported.get(variant)
        if reason:
            parser.error(f"--variants {variant} with --recipe {recipe_id}: {reason}")
    backends = BACKENDS[args.backend]
    device = torch.device("cuda")

    tensors: List[Tuple[str, str, str, torch.Tensor]] = []  # name, family, module, t
    if args.synthetic:
        for activation in args.activation:
            grads = synthetic_mlp_grads(
                activation,
                tokens=args.tokens,
                hidden=args.hidden,
                ffn=args.ffn,
                seed=args.seed,
                device=device,
            )
            for module, tensor in sorted(grads.items()):
                # family carries the activation *and* the layer, because that
                # pair is the unit of comparison here; classify() plays the same
                # role for dumps, where the module name already implies both.
                tensors.append(
                    (f"{activation}_{module}", f"{activation}/{module}", module, tensor)
                )
    elif args.tensor_path:
        from dsv3_dumps import load_dump_tensor

        name = os.path.splitext(os.path.basename(args.tensor_path))[0]
        tensors.append((name, "dump", name, load_dump_tensor(args.tensor_path)))
    else:
        from dsv3_dumps import classify, discover_dsv3_tensors, load_dump_tensor

        infos = discover_dsv3_tensors(
            args.base_dir,
            args.tensor_type,
            rank=args.rank,
            step=args.step,
            layer_names=args.layer_types,
            skip_layer_numbers=args.skip_layer_numbers,
            exclude_experts=args.exclude_experts,
        )
        print(f"Discovered {len(infos)} {args.tensor_type} tensors in {args.base_dir}")
        for info in infos:
            module = info.module_name
            name = f"layer{info.layer_num}_{module}"
            if info.expert_num is not None:
                name += f"_expert{info.expert_num}"
            tensors.append(
                (name, classify(module), module, load_dump_tensor(info.filepath))
            )

    rows: List[Dict[str, object]] = []
    for name, family, module, tensor in tensors:
        # flatten_to_2d then .cuda(), at the dump's own dtype -- the same
        # preparation plot_bias_heatmaps.py does. Deliberately no cast to
        # bfloat16: both quantizers accept fp32, and rounding an fp32 dump down
        # first would change the very quantity being measured.
        tensor = flatten_to_2d(tensor).to(device)
        for variant in args.variants:
            rows.append(
                analyze_one(
                    tensor,
                    name=name,
                    family=family,
                    module=module,
                    recipe_id=recipe_id,
                    variant=variant,
                    tensor_type=args.tensor_type,
                    backends=backends,
                    use_sr=args.use_sr,
                    seed=args.seed,
                )
            )
        del tensor
        torch.cuda.empty_cache()

    print(
        f"\nrecipe {recipe_id}, {args.tensor_type} tensors, "
        f"{'SR' if args.use_sr else 'RNE'} — medians per group\n"
    )
    print_table(rows, args.group_by)
    print(
        "\nraw0% = exact zeros as dumped (pre-rotation) · flush% = nonzero in, "
        "FP4 zero out\nfp4_0% = all FP4 zeros · p50_rel = median |x| / block_amax "
        f"(flush threshold {FP4_ZERO_RATIO:.4f})"
    )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nExported: {args.csv}")


if __name__ == "__main__":
    main()
