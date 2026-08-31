#!/usr/bin/env python3
"""Trial-averaged NVFP4 reconstruction MSE by within-block magnitude rank.

WHAT THIS ANSWERS. Stochastic rounding is unbiased by construction, so averaging
T independent QDQ trials should drive the reconstruction error down as 1/T. If it
does not -- if a magnitude bucket flattens out -- that bucket carries a
*systematic* bias that no amount of averaging removes, and a biased gradient is
exactly the kind of thing that shows up as a converged-loss gap. E03 finished
+0.0503 nat over bf16; this is a way to ask which elements paid for it.

The statistic is ported from kitchen's experimental/tensor_dump_analysis/
analyze_rank_bias.py so the two are directly comparable: rank every element
within its 16-wide quantization block by |x| (rank 1 = the block amax), average
the reconstruction over T trials, and report MSE per rank bucket against the
original. Exact zeros get their own bucket. A recipe whose SR is working traces
the dotted slope -1 guide; a bias floor is a flat tail.

WHAT IS DIFFERENT FROM KITCHEN, AND WHY. Kitchen's script measures *kitchen's*
recipes (9004, 6001, ...), which is a different quantizer from the one this
campaign trains with, and reaching it needs kitchen's two CUDA extensions plus
the psx-formats submodule -- which is access-gated. This measures torchao's
NVFP4, which is what E01-E17 actually ran, is already installed in the container
we build, and needs nothing new.

Nothing here reimplements NVFP4 numerics. Both halves are torchao's own code:

  quantize   torchao ... hadamard_quantize_row_col_{triton,cutedsl}, the same
             kernel NVFP4Linear calls in training, with stochastic_rounding=True
             and a fresh Philox seed per trial.
  dequantize torchao ... NVFP4Tensor(...).dequantize(), the formulation torchao's
             own tests use (test/prototype/moe_training/nvfp4_training/
             _assertions.py:62).

IDENTITY PATH ONLY, deliberately. The kernel emits both a row-wise and a
column-wise quantization per call. The ROW path blocks along the last dim in
1x16 tiles and reconstructs the ORIGINAL tensor -- torchao's own test asserts
exactly that (test_hadamard_quantize_row_col.py:439-441, "Rowwise SQNR:
dequantized should reconstruct raw A"). The COL path reconstructs RHT(A.T), i.e.
it lives in the Hadamard basis, so its residual is not comparable to kitchen's
`.T` variants without an inverse transform. Rather than ship a number that looks
comparable and is not, the `.T` variants are rejected; see --variant.

It lives inside the torchao package, next to the kernels it measures, so that
the two move under one revision -- the entry points here are prototype APIs and
have already been renamed once. The build that ships torchao ships this, and the
pipeline's TORCHAO_REVISION records the script's version for free.

Usage (one GPU, anywhere torchao is installed):

    python3 -m torchao.prototype.moe_training.nvfp4_training.rank_bias \
        --tensor-path /dump/G_rank0_layer26_grouped_mlp_fc2_expert0_step0.pt \
        --trials 2 4 8 16 32 64 128 --save-plot-dir /dump/rank_bias
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

# The 16-element RHT sign vector. Imported rather than copied so a change upstream
# cannot leave this measuring a basis training does not use. It only affects the
# column path's amax; the row path this script measures is RHT-free.
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
)

COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

BLOCK_SIZE = 16  # NVFP4 quantization block; not a knob, it is the format


def _load_kernels(kernel: str):
    """Return (rht_amax, rht_quantize_row_col) for the selected backend."""
    # The amax and quantize halves live in different modules -- amax in
    # hadamard_amax_*, quantize in hadamard_quantize_row_col_*.
    if kernel == "triton":
        from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
            triton_rht_amax as amax_fn,
        )
        from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (  # noqa: E501
            triton_rht_quantize_row_col as quant_fn,
        )
        return amax_fn, quant_fn
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
        cutedsl_rht_amax as amax_fn,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (  # noqa: E501
        cutedsl_rht_quantize_row_col as quant_fn,
    )
    return amax_fn, quant_fn


def _dequantize(codes: torch.Tensor, scales: torch.Tensor, amax: torch.Tensor):
    """torchao's own NVFP4 dequantize, as used by its tests."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    return (
        NVFP4Tensor(
            codes.contiguous(),
            scales.contiguous(),
            BLOCK_SIZE,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(amax),
            is_swizzled_scales=True,
        )
        .dequantize()
        .float()
    )


def make_rank_labels(tensor: torch.Tensor) -> torch.Tensor:
    """zero -> 0, nonzeros ranked 1..BLOCK_SIZE by |x| within each 1x16 block.

    Ported from kitchen's analyze_rank_bias.make_rank_labels (identity path). The
    stable sort makes equal-magnitude ties deterministic by original position, so
    two runs bucket the same elements the same way.
    """
    rows, cols = tensor.shape
    if cols % BLOCK_SIZE:
        raise ValueError(f"last dimension {cols} is not divisible by {BLOCK_SIZE}")
    blocks = tensor.reshape(rows, cols // BLOCK_SIZE, BLOCK_SIZE)
    order = torch.argsort(blocks.abs(), dim=-1, descending=True, stable=True)
    ordinal = (
        torch.arange(1, BLOCK_SIZE + 1, dtype=torch.int16, device=tensor.device)
        .view(1, 1, BLOCK_SIZE)
        .expand_as(order)
    )
    labels = torch.zeros_like(order, dtype=torch.int16)
    labels.scatter_(-1, order, ordinal)
    labels.masked_fill_(blocks == 0, 0)
    return labels.reshape_as(tensor)


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def bucket_names(occupied: List[int]) -> Dict[int, str]:
    names = {0: "zeros"}
    for rank in occupied:
        if rank == 0:
            continue
        names[rank] = "amax" if rank == 1 else f"{_ordinal(rank)} |x|"
    return names


def qdq_once(tensor, amax_fn, quant_fn, *, use_fast_math: bool, stochastic: bool):
    """One NVFP4 quantize-dequantize round trip on the identity (row) path.

    Fresh Philox bases per call are what make trials independent -- this is the
    per-trial randomness the whole statistic rests on. The row stream xors its
    seed with 1 to decorrelate it from the column stream, matching what
    nvfp4_linear does.
    """
    sign_vector = list(DEFAULT_SIGN_VECTOR)
    col_amax, row_amax = amax_fn(tensor, sign_vector)
    sr_kwargs = {}
    if stochastic:
        seed = torch.randint(
            0, 2**62, (1,), dtype=torch.int64, device=tensor.device
        )
        offset = torch.randint(
            0, 2**62, (1,), dtype=torch.int64, device=tensor.device
        )
        sr_kwargs = {
            "col_seed_base": seed,
            "col_offset_base": offset,
            "row_seed_base": seed ^ 1,
            "row_offset_base": offset,
        }
    _, _, row_codes, row_sf = quant_fn(
        tensor,
        col_amax,
        row_amax,
        sign_vector,
        stochastic_rounding=stochastic,
        use_fast_math=use_fast_math,
        **sr_kwargs,
    )
    return _dequantize(row_codes, row_sf, row_amax)


def assert_stochastic(tensor, amax_fn, quant_fn, *, use_fast_math: bool) -> None:
    """Two trials on the same input must differ.

    Without this a recipe that has quietly lost its randomness produces a
    perfectly flat curve, which reads as "no rank bias" -- the most expensive
    possible way to be wrong. Kitchen guards its sweep the same way.
    """
    a = qdq_once(tensor, amax_fn, quant_fn, use_fast_math=use_fast_math, stochastic=True)
    b = qdq_once(tensor, amax_fn, quant_fn, use_fast_math=use_fast_math, stochastic=True)
    if torch.equal(a, b):
        raise RuntimeError(
            "two stochastic trials produced identical reconstructions -- SR is not "
            "varying per trial, so every averaged curve below would be flat for the "
            "wrong reason. Check that stochastic_rounding reached the kernel."
        )


def sweep(
    tensor: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    checkpoints: List[int],
    replicas: int,
    amax_fn,
    quant_fn,
    *,
    use_fast_math: bool,
) -> Dict[int, Dict[int, float]]:
    """checkpoint -> bucket -> MSE of the trial-mean reconstruction."""
    target = tensor.float()
    labels_flat = labels.reshape(-1).long()
    n_buckets = int(counts.numel())
    checkpoint_set = set(checkpoints)
    accum: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for replica in range(replicas):
        running_sum = torch.zeros_like(target)
        for trial in range(1, checkpoints[-1] + 1):
            running_sum.add_(
                qdq_once(
                    tensor,
                    amax_fn,
                    quant_fn,
                    use_fast_math=use_fast_math,
                    stochastic=True,
                )
            )
            if trial in checkpoint_set:
                sqerr = (running_sum / trial - target).square().reshape(-1)
                sums = torch.zeros(
                    n_buckets, dtype=torch.float64, device=tensor.device
                )
                sums.scatter_add_(0, labels_flat, sqerr.double())
                mse = sums / counts
                for bucket in range(n_buckets):
                    if counts[bucket] > 0:
                        accum[trial][bucket].append(float(mse[bucket].item()))
                del sqerr, sums, mse
        print(f"  replica {replica + 1}/{replicas} complete", flush=True)

    return {
        trial: {b: sum(v) / len(v) for b, v in per_bucket.items()}
        for trial, per_bucket in accum.items()
    }


def fit_slope(results, checkpoints, bucket) -> float:
    """Least-squares slope of log10(MSE) against log10(trials) for one bucket.

    Unbiased SR averages as 1/T, so the MSE of the trial-mean falls as 1/T and
    this slope is -1. A bucket carrying a systematic bias flattens toward 0,
    because no number of trials averages a bias away. The slope is therefore the
    single number the whole plot exists to produce.
    """
    pts = [
        (math.log10(t), math.log10(results[t][bucket]))
        for t in checkpoints
        if results[t].get(bucket, 0.0) > 0.0
    ]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den


def report_metrics(results, checkpoints, occupied, names) -> None:
    """Print per-bucket slopes, then three scalars for the CI metric scraper.

    The worst (least negative) slope is the headline: it is the flattest bucket,
    i.e. the one whose error a longer average does not remove. Zeros are excluded
    -- an exactly-zero element quantizes to zero every trial, so its bucket has no
    error to decay and its slope is meaningless.
    """
    ranked = [b for b in occupied if b != 0]
    slopes = {b: fit_slope(results, checkpoints, b) for b in ranked}
    finite = {b: v for b, v in slopes.items() if not math.isnan(v)}

    print("\nSlope of log(MSE) vs log(trials), -1 = unbiased, 0 = bias floor:")
    for b in ranked:
        print(f"  {names[b]:>12}: {slopes[b]:+.4f}")

    if not finite:
        return
    worst = max(finite, key=lambda b: finite[b])
    ordered = sorted(finite.values())
    mid = len(ordered) // 2
    median = (
        ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    )
    # Flat, greppable lines: a stdout_regex metric cannot parse the table above.
    print(f"rank_bias_worst_slope: {finite[worst]:.6f}")
    print(f"rank_bias_worst_bucket: {worst}")
    print(f"rank_bias_median_slope: {median:.6f}")


def export_csv(path, results, checkpoints, occupied, names, counts) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "bucket_label",
                "n_elements",
                "trials",
                "mse_of_trial_mean",
            ],
        )
        writer.writeheader()
        for bucket in occupied:
            for trial in checkpoints:
                writer.writerow(
                    {
                        "bucket": bucket,
                        "bucket_label": names[bucket],
                        "n_elements": int(counts[bucket].item()),
                        "trials": trial,
                        "mse_of_trial_mean": results[trial][bucket],
                    }
                )
    print(f"Exported: {path}")


def plot_results(
    path, results, checkpoints, occupied, names, counts, tensor_name, label
) -> None:
    ncols = 4
    nrows = math.ceil(len(occupied) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False
    )

    for panel, bucket in enumerate(occupied):
        ax = axes.flat[panel]
        ys = [results[t][bucket] for t in checkpoints]
        ax.plot(
            checkpoints,
            ys,
            color=COLORS[0],
            linestyle="--",
            linewidth=1.8,
            marker=MARKERS[0],
            markersize=4,
            markerfacecolor="none",
            markeredgewidth=1.2,
            label=label,
        )
        # Ideal 1/T decay anchored at the first checkpoint. Departure from this
        # guide is the entire result: a bucket that flattens has a bias floor.
        if any(y > 0 for y in ys) and ys[0] > 0:
            ax.plot(
                checkpoints,
                [ys[0] * checkpoints[0] / t for t in checkpoints],
                color="dimgray",
                linestyle=":",
                linewidth=1.5,
            )
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

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[0],
            linestyle="--",
            marker=MARKERS[0],
            markerfacecolor="none",
            label=label,
        ),
        Line2D([0], [0], color="dimgray", linestyle=":", label="slope -1 (ideal)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        f"Trial-mean NVFP4 reconstruction MSE by within-block |x| rank\n"
        f"{tensor_name} — 1x{BLOCK_SIZE} blocks — {label}",
        y=1.025,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Trial-averaged NVFP4 QDQ MSE by within-block magnitude rank."
    )
    p.add_argument("--tensor-path", required=True, help="One kitchen-format .pt file.")
    p.add_argument(
        "--variant",
        default="G",
        choices=["X", "W", "G"],
        help="Labelling only -- the quantizer is identical for all three. The "
        "transpose variants (X.T/W.T/G.T) are deliberately absent: the kernel's "
        "column path reconstructs RHT(A.T), i.e. it lives in the Hadamard basis, "
        "so its residual is not comparable to the identity path without an "
        "inverse transform.",
    )
    p.add_argument("--kernel", default="triton", choices=["triton", "cutedsl"])
    p.add_argument("--use-fast-math", action="store_true", default=True)
    p.add_argument("--no-use-fast-math", dest="use_fast_math", action="store_false")
    p.add_argument("--trials", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--save-plot-dir", required=True)
    p.add_argument("--output-stem", default=None)
    p.add_argument("--skip-stochastic-check", action="store_true")
    args = p.parse_args()

    checkpoints = sorted(set(args.trials))
    if len(checkpoints) < 2 or checkpoints[0] < 1:
        p.error("--trials needs at least two distinct positive checkpoints")
    if not torch.cuda.is_available():
        p.error("NVFP4 quantization is GPU-only; no CUDA device visible")

    amax_fn, quant_fn = _load_kernels(args.kernel)

    tensor = torch.load(args.tensor_path, map_location="cpu", weights_only=True)
    if tensor.dim() != 2:
        p.error(f"expected a 2-D tensor, got {tuple(tensor.shape)}")
    # Both backends require these for the swizzled scale layout. Every tensor in
    # the E18 dump satisfies them (dims are 2048 / 1408 / 24576), but a hand-picked
    # file might not, and the kernel's own failure is far less legible than this.
    rows, cols = tensor.shape
    if rows % 128 or cols % 128:
        p.error(
            f"NVFP4 row/col kernels need both dims divisible by 128; got "
            f"{rows} x {cols}"
        )
    tensor = tensor.to(device="cuda", dtype=torch.bfloat16).contiguous()

    labels = make_rank_labels(tensor)
    counts = torch.bincount(labels.reshape(-1).long(), minlength=BLOCK_SIZE + 1)
    occupied = [b for b in range(1, BLOCK_SIZE + 1) if counts[b] > 0]
    if counts[0] > 0:
        occupied.append(0)  # zeros are always the final panel
    names = bucket_names(occupied)
    label = f"torchao NVFP4 ({args.kernel}, fast_math={args.use_fast_math})"

    print(f"Tensor: {tuple(tensor.shape)}  dtype={tensor.dtype}")
    print(f"Variant: {args.variant} (identity path, 1x{BLOCK_SIZE} blocks)")
    print(f"Quantizer: {label}")
    print(
        "Buckets: "
        + ", ".join(f"{names[b]}={int(counts[b].item()):,}" for b in occupied)
    )

    if not args.skip_stochastic_check:
        assert_stochastic(tensor, amax_fn, quant_fn, use_fast_math=args.use_fast_math)
        print("Stochastic check: two trials differ, SR is live")

    results = sweep(
        tensor,
        labels,
        counts.cuda().double(),
        checkpoints,
        args.replicas,
        amax_fn,
        quant_fn,
        use_fast_math=args.use_fast_math,
    )

    report_metrics(results, checkpoints, occupied, names)

    os.makedirs(args.save_plot_dir, exist_ok=True)
    tensor_name = os.path.splitext(os.path.basename(args.tensor_path))[0]
    stem = args.output_stem or f"rank_bias_{tensor_name}_{args.kernel}"
    export_csv(
        os.path.join(args.save_plot_dir, stem + ".csv"),
        results,
        checkpoints,
        occupied,
        names,
        counts,
    )
    plot_results(
        os.path.join(args.save_plot_dir, stem + ".png"),
        results,
        checkpoints,
        occupied,
        names,
        counts,
        tensor_name,
        label,
    )


if __name__ == "__main__":
    main()
