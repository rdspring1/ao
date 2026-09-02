#!/usr/bin/env python3
"""Trial-averaged reconstruction MSE by within-block magnitude rank (CuTe DSL).

Standalone port of kitchen's
``experimental/tensor_dump_analysis/analyze_rank_bias.py``. The kitchen leaf QDQ
(``QuantizeOpNVFP4Emulation`` on top of the psx ``clippy`` CUDA kernels) is
replaced by :mod:`nvfp4_cutedsl`, a CuTe DSL implementation that is bitwise
identical to the PyTorch oracle in :mod:`nvfp4_reference`. Nothing here imports
kitchen, psx_formats or TransformerEngine; ``torch``, ``nvidia-cutlass-dsl`` and
``matplotlib`` are the only requirements.

The full tensor is quantized in its original layout on every trial. Rank labels
are computed once and used only to aggregate the reconstruction residual after
QDQ; they never affect quantization or scaling. The ranking axis matches the QDQ
block axis for the chosen variant: last-dim ``1xB`` blocks for the identity path
(``G`` / ``X`` / ``W``), first-dim ``Bx1`` blocks for the transpose path
(``G.T`` / ``X.T`` / ``W.T``).

For a 1x16 G block, rank 1 is the largest-|x| nonzero (the amax), rank 2 is the
second largest, and so on. Original exact-zero elements share a separate bucket.

Recipe 9004 (kitchen says 6304 is "the exact same as 9004") adds the Random
Hadamard Transform on the wgrad lane: the transpose path of X and G is rotated
in groups of 16 rows before quantization and rotated back afterwards, so the
reconstruction the buckets aggregate is still in the original basis.

It calls no torchao kernel. It ships inside the torchao package only so that the
container image carries it and one revision pins script and image together --
src/torchtitan-upstream-pjnl/Dockerfile deletes /workspace/ao after installing
it, so nothing outside find_packages() survives into the image.

Example:
    RB=torchao.prototype.moe_training.nvfp4_training.rank_bias
    python3 -m $RB --tensor-path G_rank0_layer0_fc1_step0.pt \\
        --recipes 9004 --variant G.T --trials 2 4 8 16 32 64 128 \\
        --save-plot-dir ./bias_plots

    # Both recipes as two curves per bucket, to see what the rotation buys.
    python3 -m $RB --random 4096x4096 --recipes 6302 9004 \\
        --variant G.T --trials 2 4 8 16 --save-plot-dir ./bias_plots
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import torch

from . import eden_cutedsl
from . import eden_reference
from . import nvfp4_cutedsl
from . import nvfp4_reference
from . import rht

COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# backend -> quantizer -> module. A recipe picks the quantizer per tensor type
# (kitchen's QuantizeOpCustomizeTensors); the CLI picks the backend.
BACKENDS = {
    "cutedsl": {"nvfp4": nvfp4_cutedsl, "eden": eden_cutedsl},
    "torch": {"nvfp4": nvfp4_reference, "eden": eden_reference},
}

# variant -> (tensor type, transpose path)
VARIANTS = {
    "X": ("X", False),
    "X.T": ("X", True),
    "W": ("W", False),
    "W.T": ("W", True),
    "G": ("G", False),
    "G.T": ("G", True),
}

# (tensor type, transpose lane) -> GEMM the lane feeds, i.e. which
# perform_hadamard_transform_* flag and which sign vector apply. This is
# kitchen's get_gemm_types_for_tensor.
GEMM_TYPES = {
    ("X", False): "fprop",
    ("X", True): "wgrad",
    ("W", False): "fprop",
    ("W", True): "dgrad",
    ("G", False): "dgrad",
    ("G", True): "wgrad",
}

_W_2D_TILES = (
    "recipe 9004 quantizes W with 16x16 PER_2D_BLOCK tiles, which this port does "
    "not implement (only the 1x16 path is). Use --variant G / G.T / X / X.T, or "
    "run W under 6302"
)

_V2_G_ONLY = (
    "recipe 100483 is modelled for the gradient lanes only. G / G.T are the "
    "MS-EDEN operands and the reason the recipe exists; X, X.T and W.T would "
    "additionally need the psx 1x16 cast under RHT-128 (X.T) and the lazy "
    "col_rht_requantize path (W.T), neither of which is implemented here. Use "
    "--variant G / G.T"
)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """The axes of a kitchen recipe this port can express.

    ``use_sr``: tensor type -> stochastic rounding, applied to both lanes of that
    tensor (kitchen's ``use_sr``). Under an EDEN quantizer this names the
    rounding of the *block scale*, not of the data, which is always RNE there.
    ``quantizer``: tensor type -> leaf quantizer, mirroring kitchen's
    ``QuantizeOpCustomizeTensors``.
    ``rht_gemms``: the GEMMs the recipe rotates, i.e. which
    ``perform_hadamard_transform_{fprop,wgrad,dgrad}`` flags are set. A lane
    rotates when ``GEMM_TYPES[(tensor_type, transpose)]`` is in this set, which
    is how kitchen decides it (``get_rht_settings_for_tensor``), and the sign
    vector is the one checked in for that GEMM.
    ``dynamic_signs``: kitchen's ``enable_online_randomization``. 9004 leaves it
    off, so its sign vector is fixed for the whole run and the checked-in vector
    reproduces it bitwise. 100483 turns it on, so a real run re-draws the vector
    every iteration -- and because that is what makes the estimator unbiased, a
    9004-style frozen vector does not merely approximate 100483, it changes the
    answer (the MSE-vs-trials slope goes from -1.0 to -0.02). Sweeping both
    recipes in one run therefore has to vary the sign for one and freeze it for
    the other, which is why this is a recipe property and not just a CLI flag.
    ``unsupported``: variant -> why this recipe cannot be swept for it.
    """

    use_sr: Dict[str, bool]
    quantizer: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {"X": "nvfp4", "W": "nvfp4", "G": "nvfp4"}
    )
    rht_dim: int = 16
    rht_gemms: FrozenSet[str] = frozenset()
    dynamic_signs: bool = False
    unsupported: Dict[str, str] = dataclasses.field(default_factory=dict)


# 6302 is kitchen's ``QuantizeRecipe.NVFP4_EMULATION`` with ``use_sr`` on
# ``g_params``; 9004 is 6302 plus 16x16 W tiles and the wgrad RHT; ``nvfp4`` is
# the base recipe untouched, i.e. RNE everywhere. 100483 replaces the gradient
# quantizer with MS-EDEN and rotates at dim 128 on *both* backward GEMMs.
# Subchannel scaling, UE5M3 scales, 1x32 tiles and the 2D-block tile shapes are
# still not modelled -- see README.md.
RECIPES: Dict[str, Recipe] = {
    "6302": Recipe(use_sr={"X": False, "W": False, "G": True}),
    "9004": Recipe(
        use_sr={"X": False, "W": False, "G": True},
        rht_dim=16,
        rht_gemms=frozenset({"wgrad"}),
        dynamic_signs=False,  # enable_online_randomization off
        unsupported={"W": _W_2D_TILES, "W.T": _W_2D_TILES},
    ),
    "100483": Recipe(
        # QuantizeOpEdenFP4Emulation(stochastic_round_scale=True) on G only; X
        # and W keep the psx NVFP4 quantizer.
        use_sr={"X": False, "W": False, "G": True},
        quantizer={"X": "nvfp4", "W": "nvfp4", "G": "eden"},
        rht_dim=128,
        # _create_rht_technique_recipe(rht_size=128, all_backward=False): wgrad
        # and dgrad both on, fprop off.
        rht_gemms=frozenset({"wgrad", "dgrad"}),
        dynamic_signs=True,  # enable_online_randomization on
        unsupported={
            "X": _V2_G_ONLY,
            "X.T": _V2_G_ONLY,
            "W": _V2_G_ONLY,
            "W.T": _V2_G_ONLY,
        },
    ),
    "nvfp4": Recipe(use_sr={"X": False, "W": False, "G": False}),
}

# kitchen/config.py:2270: "this recipe is the exact same as 9004".
RECIPE_ALIASES = {"6304": "9004"}


def flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Collapse leading dims so a [..., D] tensor becomes [prod(...), D]."""
    if t.dim() > 2:
        return t.reshape(-1, t.shape[-1])
    return t


def make_rank_labels(
    tensor: torch.Tensor, block_size: int, *, transpose: bool = False
) -> torch.Tensor:
    """Return labels shaped like tensor: zero=0, nonzeros ranked 1..block_size.

    Ranking is within each QDQ block. Identity-path variants block along the
    last dim (``1xB``); transpose-path variants block along the first dim
    (``Bx1``), matching the transpose quantization path.
    """
    tensor = flatten_to_2d(tensor)
    rows, cols = tensor.shape
    if transpose:
        if rows % block_size:
            raise ValueError(
                f"first dimension {rows} is not divisible by block size {block_size} "
                f"(required for transpose-path ranking)"
            )
        blocks = tensor.reshape(rows // block_size, block_size, cols)
        rank_dim = 1
    else:
        if cols % block_size:
            raise ValueError(
                f"last dimension {cols} is not divisible by block size {block_size}"
            )
        blocks = tensor.reshape(rows, cols // block_size, block_size)
        rank_dim = -1

    # Stable sort makes equal-magnitude ties deterministic by original position.
    order = torch.argsort(blocks.abs(), dim=rank_dim, descending=True, stable=True)
    ordinal_shape = [1, 1, 1]
    ordinal_shape[rank_dim if rank_dim >= 0 else 2] = block_size
    ordinal = (
        torch.arange(1, block_size + 1, dtype=torch.int16)
        .view(*ordinal_shape)
        .expand_as(order)
    )
    labels = torch.zeros_like(order, dtype=torch.int16)
    labels.scatter_(rank_dim, order, ordinal)
    labels.masked_fill_(blocks == 0, 0)
    return labels.reshape_as(tensor)


def _ordinal(n: int) -> str:
    """English ordinal suffix: 1st, 2nd, 3rd, 4th, ..., 21st, 22nd, 23rd, 24th."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def bucket_names(occupied: List[int]) -> Dict[int, str]:
    names = {0: "zeros"}
    for rank in occupied:
        if rank == 0:
            continue  # keep "zeros"; do not rename to "0th |x|"
        if rank == 1:
            names[rank] = "amax"
        else:
            names[rank] = f"{_ordinal(rank)} |x|"
    return names


def trial_seed(base_seed: int, replica: int, trial: int, max_trials: int) -> int:
    """Distinct Philox stream per (replica, trial)."""
    return base_seed + replica * (max_trials + 1) + trial


RHTMatrices = Tuple[torch.Tensor, torch.Tensor]


def rht_matrices(
    device: torch.device, sign_seed: Optional[int], dim: int, gemm: str
) -> RHTMatrices:
    """Forward and inverse rotation for one lane; ``sign_seed=None`` is kitchen's
    checked-in sign vector for that GEMM."""
    return rht.transform_matrices(
        rht.sign_vector(device, seed=sign_seed, dim=dim, lane=gemm)
    )


def leaf_qdq(
    backend,
    tensor: torch.Tensor,
    *,
    transpose: bool,
    use_sr: bool,
    matrices: Optional[RHTMatrices],
    rht_dim: int,
    seed: int,
) -> torch.Tensor:
    """One trial's QDQ, rotated into the lane's basis first when the recipe asks.

    Kitchen rotates, quantizes, inverse-rotates and only then crops the RHT
    padding (``QuantizeOpFP8HadamardTransform.dequantize``); the order matters
    because the inverse mixes the padded elements back in.

    The rotation runs along the lane's own contraction axis -- rows for a
    transpose (wgrad) lane, columns for an identity (dgrad) lane -- which is the
    axis the QDQ blocks along, so an RHT tile and a quantization block cover the
    same elements.
    """
    if matrices is None:
        return backend.quant_dequant(
            tensor, transpose=transpose, use_sr=use_sr, seed=seed
        )
    forward, inverse = matrices
    padded = (
        rht.pad_rows(tensor, rht_dim) if transpose else rht.pad_cols(tensor, rht_dim)
    )
    rotated = rht.transform(padded, forward, transpose=transpose)
    dq = backend.quant_dequant(rotated, transpose=transpose, use_sr=use_sr, seed=seed)
    restored = rht.transform(dq, inverse, transpose=transpose)
    return restored[: tensor.shape[0], : tensor.shape[1]]


def assert_stochastic(
    backend,
    use_sr: bool,
    transpose: bool,
    matrices: Optional[RHTMatrices],
    rht_dim: int,
) -> None:
    """Two trials on the same tensor must differ, else trials aren't independent."""
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(transpose=transpose, use_sr=use_sr, matrices=matrices, rht_dim=rht_dim)
    a = leaf_qdq(backend, x, seed=1, **kwargs)
    b = leaf_qdq(backend, x, seed=2, **kwargs)
    if torch.equal(a.float(), b.float()):
        raise RuntimeError(
            "Recipe produced identical reconstructions on two trials; the "
            "recipe is deterministic for this variant (no stochastic rounding), "
            "so trial averaging cannot reduce its bias. Re-run with "
            "--skip-stochastic-check to sweep it anyway."
        )


def sweep_recipe(
    tensor: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    use_sr: bool,
    checkpoints: List[int],
    replicas: int,
    transpose: bool,
    backend,
    base_seed: int,
    use_rht: bool,
    vary_rht_sign: bool,
    rht_dim: int,
    rht_gemm: str,
    signed_out: Optional[Dict[int, Dict[int, Tuple[float, float]]]] = None,
) -> Dict[int, Dict[int, float]]:
    """Return checkpoint -> bucket -> MSE of trial-mean reconstruction.

    ``signed_out``, when given, is filled with checkpoint -> bucket ->
    ``(signed_mean, abs_shrink)``. MSE is a squared quantity and therefore cannot
    say whether a bucket's residual error points the SAME WAY on every element.
    That distinction decides whether a bias survives into a parameter update or
    cancels: a per-element error with a random sign averages out across a tensor
    even though it never averages across trials, and only a coherent one can move
    a converged loss.

    ``signed_mean`` is the bucket mean of ``recon_mean - x``; ``abs_shrink`` is the
    bucket mean of ``|recon_mean| - |x|``. The second is the one that tests the
    saturation hypothesis directly: a block scale that rounds DOWN makes
    ``amax / block_scale > 6``, so the amax saturates and comes back strictly
    smaller in magnitude, while a scale that rounds up leaves the amax below the
    E2M1 ceiling with rounding freedom SR can use. If that is the mechanism,
    ``abs_shrink`` is systematically negative for the amax bucket and ~0 elsewhere.

    An out-parameter rather than a second return value: two call sites already
    unpack the MSE dict, and both would otherwise have to change for a statistic
    only one of them reports.
    """
    target = tensor.float()
    labels_flat = labels.reshape(-1).long()
    n_buckets = int(counts.numel())
    checkpoint_set = set(checkpoints)
    max_trials = checkpoints[-1]
    accum: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    signed_accum: Dict[str, Dict[int, List[torch.Tensor]]] = {
        "s": defaultdict(list),
        "a": defaultdict(list),
    }

    frozen = (
        rht_matrices(tensor.device, None, rht_dim, rht_gemm) if use_rht else None
    )

    for replica in range(replicas):
        running_sum = torch.zeros_like(target, dtype=torch.float32)
        for trial in range(1, max_trials + 1):
            seed = trial_seed(base_seed, replica, trial, max_trials)
            matrices = (
                rht_matrices(tensor.device, seed, rht_dim, rht_gemm)
                if use_rht and vary_rht_sign
                else frozen
            )
            recon = leaf_qdq(
                backend,
                tensor,
                transpose=transpose,
                use_sr=use_sr,
                matrices=matrices,
                rht_dim=rht_dim,
                seed=seed,
            )
            running_sum.add_(recon.float())

            if trial in checkpoint_set:
                residual = running_sum / trial - target
                sqerr = residual.square().reshape(-1)
                sums = torch.zeros(n_buckets, dtype=torch.float64, device=tensor.device)
                sums.scatter_add_(0, labels_flat, sqerr.double())
                mse_by_bucket = sums / counts
                for bucket in range(n_buckets):
                    if counts[bucket] > 0:
                        accum[trial][bucket].append(float(mse_by_bucket[bucket].item()))
                del sqerr, sums, mse_by_bucket

                if signed_out is not None:
                    shrink = (
                        (running_sum / trial).abs() - target.abs()
                    ).reshape(-1)
                    for name, flat in (("s", residual.reshape(-1)), ("a", shrink)):
                        acc = torch.zeros(
                            n_buckets, dtype=torch.float64, device=tensor.device
                        )
                        acc.scatter_add_(0, labels_flat, flat.double())
                        signed_accum[name][trial].append((acc / counts).cpu())
                    del shrink, acc
                del residual

            del recon
        print(f"  replica {replica + 1}/{replicas} complete")

    if signed_out is not None:
        for trial in signed_accum["s"]:
            s_mean = torch.stack(signed_accum["s"][trial]).mean(0)
            a_mean = torch.stack(signed_accum["a"][trial]).mean(0)
            signed_out[trial] = {
                bucket: (float(s_mean[bucket]), float(a_mean[bucket]))
                for bucket in range(n_buckets)
                if counts[bucket] > 0
            }

    return {
        trial: {
            bucket: sum(values) / len(values) for bucket, values in per_bucket.items()
        }
        for trial, per_bucket in accum.items()
    }


def report_signed_bias(
    recipe_id: str,
    signed: Dict[int, Dict[int, Tuple[float, float]]],
    checkpoints: List[int],
    occupied: List[int],
    names: Dict[int, str],
    rms_by_bucket: Dict[int, float],
) -> None:
    """Print the signed residual and the magnitude shrink at the largest T.

    A bucket's MSE says how much error there is; these say whether it points one
    way. ``coherence`` is |signed mean| / RMS residual: 1.0 means every element of
    the bucket errs in the same direction and the whole residual survives into a
    parameter update, ~0 means the signs cancel and it does not, whatever the MSE.
    """
    t = checkpoints[-1]
    per_bucket = signed.get(t, {})
    if not per_bucket:
        return
    ranked = [b for b in occupied if b != 0]

    print(f"\nrecipe {recipe_id}: signed bias at T={t} (MSE cannot see this)")
    print(f"  {'bucket':>10} {'signed mean':>13} {'|q|-|x| mean':>14} {'coherence':>10}")
    for b in ranked:
        s_mean, a_mean = per_bucket[b]
        rms = rms_by_bucket.get(b, 0.0)
        coh = abs(s_mean) / rms if rms > 0 else float("nan")
        print(f"  {names[b]:>10} {s_mean:13.4e} {a_mean:14.4e} {coh:10.4f}")

    amax_s, amax_a = per_bucket[ranked[0]]
    rms = rms_by_bucket.get(ranked[0], 0.0)
    print(f"rank_bias_{recipe_id}_amax_signed_mean: {amax_s:.6e}")
    print(f"rank_bias_{recipe_id}_amax_abs_shrink: {amax_a:.6e}")
    if rms > 0:
        print(f"rank_bias_{recipe_id}_amax_coherence: {abs(amax_s) / rms:.6f}")


def export_csv(
    path: str,
    results: Dict[str, Dict[int, Dict[int, float]]],
    checkpoints: List[int],
    occupied: List[int],
    names: Dict[int, str],
    counts: torch.Tensor,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recipe_id",
                "bucket",
                "bucket_label",
                "n_elements",
                "trials",
                "mse_of_trial_mean",
            ],
        )
        writer.writeheader()
        for recipe_id, per_trial in results.items():
            for bucket in occupied:
                for trial in checkpoints:
                    writer.writerow(
                        {
                            "recipe_id": recipe_id,
                            "bucket": bucket,
                            "bucket_label": names[bucket],
                            "n_elements": int(counts[bucket].item()),
                            "trials": trial,
                            "mse_of_trial_mean": per_trial[trial][bucket],
                        }
                    )
    print(f"Exported: {path}")


def plot_results(
    path: str,
    results: Dict[str, Dict[int, Dict[int, float]]],
    checkpoints: List[int],
    occupied: List[int],
    names: Dict[int, str],
    counts: torch.Tensor,
    tensor_name: str,
    block_size: int,
    variant: str,
    *,
    transpose: bool = False,
    rht_recipes: Optional[List[str]] = None,
) -> None:
    ncols = 4
    nrows = math.ceil(len(occupied) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False
    )

    for panel, bucket in enumerate(occupied):
        ax = axes.flat[panel]
        first_values = []
        for ri, (recipe_id, per_trial) in enumerate(results.items()):
            ys = [per_trial[t][bucket] for t in checkpoints]
            first_values.append(ys[0])
            ax.plot(
                checkpoints,
                ys,
                color=COLORS[ri % len(COLORS)],
                linestyle="--",
                linewidth=1.8,
                marker=MARKERS[ri % len(MARKERS)],
                markersize=4,
                markerfacecolor="none",
                markeredgewidth=1.2,
                label=f"recipe {recipe_id}",
            )

        # Per-panel ideal MSE decay guide, anchored at checkpoints[0]
        # (smallest trial count; T=2 for the default --trials list).
        # Skip log-y + guide when a panel is all zeros (e.g. exact-zero bucket).
        y0 = max(first_values) if first_values else 0.0
        all_ys = [
            per_trial[t][bucket] for per_trial in results.values() for t in checkpoints
        ]
        has_positive = any(y > 0 for y in all_ys)
        if has_positive and y0 > 0:
            guide = [y0 * checkpoints[0] / t for t in checkpoints]
            ax.plot(checkpoints, guide, color="dimgray", linestyle=":", linewidth=1.5)
            ax.set_yscale("log")
        else:
            ax.text(
                0.5,
                0.5,
                "no positive finite MSE\n(all 0 or NaN)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="dimgray",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(checkpoints)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(f"{names[bucket]}  (n={int(counts[bucket].item()):,})", fontsize=9)
        ax.set_xlabel("Trials T")
        ax.set_ylabel("MSE")

    for panel in range(len(occupied), nrows * ncols):
        axes.flat[panel].set_visible(False)

    recipe_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[i % len(COLORS)],
            linestyle="--",
            marker=MARKERS[i % len(MARKERS)],
            markerfacecolor="none",
            label=f"recipe {recipe_id}",
        )
        for i, recipe_id in enumerate(results)
    ]
    recipe_handles.append(
        Line2D([0], [0], color="dimgray", linestyle=":", label="slope -1 (ideal)")
    )
    fig.legend(
        handles=recipe_handles,
        loc="upper center",
        ncol=min(len(recipe_handles), 5),
        title="recipe",
        bbox_to_anchor=(0.5, 0.985),
    )
    block_axis = f"{block_size}x1" if transpose else f"1x{block_size}"
    # rht_recipes entries are already "<id> RHT-<dim>/<gemm>": the dimension and
    # the rotated GEMM both differ between 9004 and 100483, so neither can be
    # hard-coded into this label.
    rht_note = f" — {', '.join(rht_recipes)}" if rht_recipes else ""
    fig.suptitle(
        f"Trial-mean reconstruction MSE by within-block |x| rank\n"
        f"{tensor_name} — variant {variant} — {block_axis} QDQ blocks{rht_note}",
        y=1.025,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_tensor(args) -> torch.Tensor:
    if args.random:
        rows, cols = (int(v) for v in args.random.lower().split("x"))
        generator = torch.Generator().manual_seed(args.seed)
        # Heavy-tailed like a real gradient dump, so the rank buckets are not
        # all statistically identical.
        normal = torch.randn(rows, cols, generator=generator)
        return (normal * torch.exp(0.5 * torch.randn(rows, 1, generator=generator))).to(
            torch.bfloat16
        )
    tensor = torch.load(args.tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected a bare tensor dump, got {type(tensor)}")
    return tensor.detach()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trial-averaged full-tensor NVFP4 QDQ MSE by within-block rank."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tensor-path", help="Path to a bare-tensor .pt dump.")
    source.add_argument(
        "--random",
        metavar="ROWSxCOLS",
        help="Sweep a synthetic heavy-tailed tensor of this shape instead of a dump.",
    )
    parser.add_argument(
        "--recipes",
        nargs="+",
        default=["9004"],
        choices=sorted(set(RECIPES) | set(RECIPE_ALIASES)),
        help="Recipes to sweep, one curve each (default: 9004; 6304 is an alias "
        "for 9004).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="G",
        choices=list(VARIANTS),
        help="Leaf QDQ path: X, X.T, W, W.T, G, G.T (default: G).",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--trials", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="cutedsl",
        help="QDQ implementation (both are bitwise identical; default: cutedsl).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base Philox seed.")
    parser.add_argument("--save-plot-dir", required=True)
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Basename (no extension) for the .png/.csv written into "
        "--save-plot-dir. Defaults to a name derived from the tensor, "
        "variant, block size and recipes.",
    )
    parser.add_argument(
        "--vary-rht-sign",
        dest="vary_rht_sign",
        action="store_true",
        default=None,
        help="Re-draw the RHT sign vector every trial. The default follows the "
        "recipe: frozen for 9004 (enable_online_randomization off, so the "
        "checked-in vector is what a run actually uses and the only setting "
        "that reproduces it bitwise), re-drawn for 100483 (online "
        "randomization on). Passing this flag, or --no-vary-rht-sign, forces "
        "the same choice on every recipe in the run -- which is what you want "
        "to isolate the effect of the sign lifetime, and not what you want for "
        "a faithful 9004-vs-100483 comparison.",
    )
    parser.add_argument(
        "--no-vary-rht-sign", dest="vary_rht_sign", action="store_false"
    )
    parser.add_argument(
        "--skip-stochastic-check",
        action="store_true",
        help="Skip the pre-flight check that trials are independent.",
    )
    args = parser.parse_args()

    checkpoints = sorted(set(args.trials))
    if len(checkpoints) < 2 or checkpoints[0] < 1:
        parser.error("--trials needs at least two distinct positive checkpoints")
    if args.replicas < 1:
        parser.error("--replicas must be >= 1")
    if args.block_size != nvfp4_reference.BLOCK_SIZE:
        parser.error(
            f"--block-size must be {nvfp4_reference.BLOCK_SIZE}: the QDQ path is "
            "NVFP4 with 1x16 quantization tiles, and the rank buckets must line "
            "up with the quantization blocks."
        )

    backends = BACKENDS[args.backend]
    tensor_type, transpose = VARIANTS[args.variant]
    gemm = GEMM_TYPES[(tensor_type, transpose)]
    recipe_ids = [RECIPE_ALIASES.get(r, r) for r in args.recipes]
    for recipe_id in recipe_ids:
        reason = RECIPES[recipe_id].unsupported.get(args.variant)
        if reason:
            parser.error(f"--variant {args.variant} with --recipes {recipe_id}: {reason}")
    tensor_cpu = flatten_to_2d(load_tensor(args))
    labels_cpu = make_rank_labels(tensor_cpu, args.block_size, transpose=transpose)
    counts_cpu = torch.bincount(
        labels_cpu.reshape(-1).long(), minlength=args.block_size + 1
    )
    occupied = [
        bucket for bucket in range(1, args.block_size + 1) if counts_cpu[bucket] > 0
    ]
    if counts_cpu[0] > 0:
        occupied.append(0)  # zeros are always the final subplot
    names = bucket_names(occupied)

    print(f"Tensor: {tuple(tensor_cpu.shape)} {tensor_cpu.dtype}")
    print(f"Variant: {args.variant} (tensor_type={tensor_type}, transpose={transpose})")
    print(f"Backend: {args.backend}, base seed {args.seed}")
    print(
        "Buckets: "
        + ", ".join(f"{names[b]}={int(counts_cpu[b].item()):,}" for b in occupied)
    )

    # A lane rotates when the GEMM it feeds is one the recipe transforms, which
    # is how kitchen decides it. 9004 sets wgrad only, so its identity lanes are
    # all unrotated; 100483 sets wgrad and dgrad, so both G lanes rotate -- and
    # they draw different sign vectors, because they are different GEMMs.
    settings = {
        recipe_id: (
            RECIPES[recipe_id].use_sr[tensor_type],
            gemm in RECIPES[recipe_id].rht_gemms,
            RECIPES[recipe_id].rht_dim,
            backends[RECIPES[recipe_id].quantizer[tensor_type]],
            (
                RECIPES[recipe_id].dynamic_signs
                if args.vary_rht_sign is None
                else args.vary_rht_sign
            ),
        )
        for recipe_id in recipe_ids
    }
    if not args.skip_stochastic_check:
        for use_sr, use_rht, rht_dim, backend, _ in settings.values():
            assert_stochastic(
                backend,
                use_sr,
                transpose,
                (
                    rht_matrices(torch.device("cuda"), None, rht_dim, gemm)
                    if use_rht
                    else None
                ),
                rht_dim,
            )

    tensor = tensor_cpu.cuda()
    labels = labels_cpu.cuda()
    counts = counts_cpu.cuda().double()
    results = {}
    signed: Dict[str, Dict[int, Dict[int, Tuple[float, float]]]] = {}
    for recipe_id, (use_sr, use_rht, rht_dim, backend, vary) in settings.items():
        sign = "varying" if vary else "fixed"
        rht_note = f", RHT-{rht_dim} on {gemm} (sign {sign})" if use_rht else ""
        quantizer = RECIPES[recipe_id].quantizer[tensor_type]
        sr_note = "scale SR" if quantizer == "eden" else "use_sr"
        print(f"recipe {recipe_id}: {quantizer}, {sr_note}={use_sr}{rht_note}")
        signed[recipe_id] = {}
        results[recipe_id] = sweep_recipe(
            tensor,
            labels,
            counts,
            use_sr,
            checkpoints,
            args.replicas,
            transpose,
            backend,
            args.seed,
            use_rht,
            vary,
            rht_dim,
            gemm,
            signed_out=signed[recipe_id],
        )
        torch.cuda.empty_cache()

    for recipe_id, per_trial in results.items():
        rms = {
            b: math.sqrt(per_trial[checkpoints[-1]][b])
            for b in per_trial[checkpoints[-1]]
        }
        report_signed_bias(
            recipe_id, signed[recipe_id], checkpoints, occupied, names, rms
        )

    os.makedirs(args.save_plot_dir, exist_ok=True)
    if args.tensor_path:
        tensor_name = os.path.splitext(os.path.basename(args.tensor_path))[0]
    else:
        tensor_name = f"random_{args.random}"
    recipe_tag = "_".join(recipe_ids)
    variant_tag = args.variant.replace(".", "t")
    stem = args.output_stem or (
        f"rank_bias_{tensor_name}_{variant_tag}_b{args.block_size}_{recipe_tag}"
    )
    export_csv(
        os.path.join(args.save_plot_dir, stem + ".csv"),
        results,
        checkpoints,
        occupied,
        names,
        counts_cpu,
    )
    plot_results(
        os.path.join(args.save_plot_dir, stem + ".png"),
        results,
        checkpoints,
        occupied,
        names,
        counts_cpu,
        tensor_name,
        args.block_size,
        args.variant,
        transpose=transpose,
        rht_recipes=[
            f"{r} RHT-{dim}/{gemm}"
            for r, (_, use_rht, dim, _, _) in settings.items()
            if use_rht
        ],
    )


if __name__ == "__main__":
    main()
