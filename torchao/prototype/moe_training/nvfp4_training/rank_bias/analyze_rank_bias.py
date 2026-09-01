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

from . import nvfp4_cutedsl
from . import nvfp4_reference
from . import rht

COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

BACKENDS = {"cutedsl": nvfp4_cutedsl, "torch": nvfp4_reference}

# variant -> (tensor type, transpose path)
VARIANTS = {
    "X": ("X", False),
    "X.T": ("X", True),
    "W": ("W", False),
    "W.T": ("W", True),
    "G": ("G", False),
    "G.T": ("G", True),
}

_W_2D_TILES = (
    "recipe 9004 quantizes W with 16x16 PER_2D_BLOCK tiles, which this port does "
    "not implement (only the 1x16 path is). Use --variant G / G.T / X / X.T, or "
    "run W under 6302"
)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """The axes of a kitchen recipe this port can express.

    ``use_sr``: tensor type -> stochastic rounding (kitchen's ``use_sr``, which
    applies to both the identity and transpose lane of that tensor).
    ``rht_transpose``: tensor types whose *transpose* lane is rotated by the
    wgrad RHT. Kitchen never rotates an identity lane in these recipes and never
    rotates W at all.
    ``unsupported``: variant -> why this recipe cannot be swept for it.
    """

    use_sr: Dict[str, bool]
    rht_transpose: FrozenSet[str] = frozenset()
    unsupported: Dict[str, str] = dataclasses.field(default_factory=dict)


# 6302 is kitchen's ``QuantizeRecipe.NVFP4_EMULATION`` with ``use_sr`` on
# ``g_params``; 9004 is 6302 plus 16x16 W tiles and the wgrad RHT; ``nvfp4`` is
# the base recipe untouched, i.e. RNE everywhere. EDEN, UE5M3 scales, 1x32 tiles
# and 2-level subchannel scaling are not modelled -- see README.md.
RECIPES: Dict[str, Recipe] = {
    "6302": Recipe(use_sr={"X": False, "W": False, "G": True}),
    "9004": Recipe(
        use_sr={"X": False, "W": False, "G": True},
        rht_transpose=frozenset({"X", "G"}),
        unsupported={"W": _W_2D_TILES, "W.T": _W_2D_TILES},
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


def rht_matrices(device: torch.device, sign_seed: Optional[int]) -> RHTMatrices:
    """Forward and inverse wgrad rotation; ``sign_seed=None`` is kitchen's fixed sign."""
    return rht.transform_matrices(rht.sign_vector(device, seed=sign_seed))


def leaf_qdq(
    backend,
    tensor: torch.Tensor,
    *,
    transpose: bool,
    use_sr: bool,
    matrices: Optional[RHTMatrices],
    seed: int,
) -> torch.Tensor:
    """One trial's QDQ, rotated into the wgrad basis first when the recipe asks.

    Kitchen rotates, quantizes, inverse-rotates and only then crops the RHT row
    padding (``QuantizeOpFP8HadamardTransform.dequantize``); the order matters
    because the inverse mixes the padded rows back in.
    """
    if matrices is None:
        return backend.quant_dequant(
            tensor, transpose=transpose, use_sr=use_sr, seed=seed
        )
    forward, inverse = matrices
    rotated = rht.transform(rht.pad_rows(tensor), forward)
    dq = backend.quant_dequant(rotated, transpose=True, use_sr=use_sr, seed=seed)
    return rht.transform(dq, inverse)[: tensor.shape[0]]


def assert_stochastic(
    backend, use_sr: bool, transpose: bool, matrices: Optional[RHTMatrices]
) -> None:
    """Two trials on the same tensor must differ, else trials aren't independent."""
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    a = leaf_qdq(backend, x, transpose=transpose, use_sr=use_sr, matrices=matrices, seed=1)
    b = leaf_qdq(backend, x, transpose=transpose, use_sr=use_sr, matrices=matrices, seed=2)
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

    frozen = rht_matrices(tensor.device, None) if use_rht else None

    for replica in range(replicas):
        running_sum = torch.zeros_like(target, dtype=torch.float32)
        for trial in range(1, max_trials + 1):
            seed = trial_seed(base_seed, replica, trial, max_trials)
            matrices = (
                rht_matrices(tensor.device, seed)
                if use_rht and vary_rht_sign
                else frozen
            )
            recon = leaf_qdq(
                backend,
                tensor,
                transpose=transpose,
                use_sr=use_sr,
                matrices=matrices,
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


def fit_slope(per_trial, checkpoints, bucket) -> float:
    """Least-squares slope of log10(MSE) against log10(trials) for one bucket.

    Unbiased SR averages as 1/T, so the MSE of the trial-mean falls as 1/T and
    this slope is -1. A bucket carrying a systematic bias flattens toward 0,
    because no number of trials averages a bias away. The slope is therefore the
    single number the whole plot exists to produce.
    """
    pts = [
        (math.log10(t), math.log10(per_trial[t][bucket]))
        for t in checkpoints
        if per_trial[t].get(bucket, 0.0) > 0.0
    ]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den


def report_metrics(recipe_id, per_trial, checkpoints, occupied, names) -> None:
    """Print per-bucket slopes, then three scalars for the CI metric scraper.

    The worst (least negative) slope is the headline: it is the flattest bucket,
    i.e. the one whose error a longer average does not remove. Zeros are excluded
    -- an exactly-zero element quantizes to zero every trial, so its bucket has no
    error to decay and its slope is meaningless.

    The recipe id is IN the metric name rather than a separate line, because a
    JET ``stdout_regex`` metric has no way to associate two lines with each other
    and ``--recipes`` may sweep several in one run.
    """
    ranked = [b for b in occupied if b != 0]
    slopes = {b: fit_slope(per_trial, checkpoints, b) for b in ranked}
    finite = {b: v for b, v in slopes.items() if not math.isnan(v)}

    print(f"\nrecipe {recipe_id}: slope of log(MSE) vs log(trials), "
          "-1 = unbiased, 0 = bias floor:")
    for b in ranked:
        print(f"  {names[b]:>12}: {slopes[b]:+.4f}")

    if not finite:
        return
    worst = max(finite, key=lambda b: finite[b])
    ordered = sorted(finite.values())
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    # Flat, greppable lines: a stdout_regex metric cannot parse the table above.
    print(f"rank_bias_{recipe_id}_worst_slope: {finite[worst]:.6f}")
    print(f"rank_bias_{recipe_id}_worst_bucket: {worst}")
    print(f"rank_bias_{recipe_id}_median_slope: {median:.6f}")


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
    rht_note = (
        f" — RHT-16 (wgrad) for {', '.join(rht_recipes)}" if rht_recipes else ""
    )
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
        default=False,
        help="Re-draw the RHT sign vector every trial, for dumps whose sign "
        "vector was randomized online. The default freezes it to kitchen's "
        "checked-in wgrad vector, which is what recipe 9004 itself uses and the "
        "only setting that reproduces a kitchen run bitwise.",
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

    backend = BACKENDS[args.backend]
    tensor_type, transpose = VARIANTS[args.variant]
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

    # RHT only ever touches a transpose lane, and only for the tensor types the
    # recipe rotates (never W).
    settings = {
        recipe_id: (
            RECIPES[recipe_id].use_sr[tensor_type],
            transpose and tensor_type in RECIPES[recipe_id].rht_transpose,
        )
        for recipe_id in recipe_ids
    }
    if not args.skip_stochastic_check:
        for use_sr, use_rht in settings.values():
            assert_stochastic(
                backend,
                use_sr,
                transpose,
                rht_matrices(torch.device("cuda"), None) if use_rht else None,
            )

    tensor = tensor_cpu.cuda()
    labels = labels_cpu.cuda()
    counts = counts_cpu.cuda().double()
    results = {}
    signed: Dict[str, Dict[int, Dict[int, Tuple[float, float]]]] = {}
    for recipe_id, (use_sr, use_rht) in settings.items():
        sign = "varying" if args.vary_rht_sign else "fixed"
        rht_note = f", RHT-16 (sign {sign})" if use_rht else ""
        print(f"recipe {recipe_id}: use_sr={use_sr}{rht_note}")
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
            args.vary_rht_sign,
            signed_out=signed[recipe_id],
        )
        torch.cuda.empty_cache()

    for recipe_id, per_trial in results.items():
        report_metrics(recipe_id, per_trial, checkpoints, occupied, names)
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
        rht_recipes=[r for r, (_, use_rht) in settings.items() if use_rht],
    )


if __name__ == "__main__":
    main()
