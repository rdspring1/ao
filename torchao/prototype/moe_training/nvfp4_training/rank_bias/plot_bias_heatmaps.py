#!/usr/bin/env python3
"""Batch rank-bias analysis plus summary heatmaps, for DSV3 tensor dumps.

Port of kitchen's ``experimental/tensor_dump_analysis/plot_bias_heatmaps.py``,
adapted to this repo's kitchen-free ``analyze_rank_bias.py`` (local
``RECIPES``/``sweep_recipe``, not kitchen's ``QLinearParams`` dispatch) and to
DSV3's dump filename convention and layer layout (no mamba layers; see
``dsv3_dumps.py``).

Default mode preserves kitchen's old behavior: read a flatness summary CSV and
plot a slope heatmap. With ``--run-rank-bias``, this script becomes the
top-level driver: discover DSV3 tensor dumps, call ``analyze_rank_bias`` for
each tensor, upsert ``_flatness_summary.csv``, then render the heatmap.

Example full workflow:
    python plot_bias_heatmaps.py \\
        --run-rank-bias \\
        --base-dir dsv3_16b_f0l0_mxfp8_attn_1500steps_layer26_fc2 \\
        --variant G.T --rank 0 --step 0 --recipes 9004 \\
        --trials 2 4 8 16 32 64 128 \\
        --out-dir ./bias_plots --tag dsv3_9004

Example plot-only workflow:
    python plot_bias_heatmaps.py \\
        --slope-csv ./bias_plots/_flatness_summary.csv \\
        --out-dir ./heatmaps

Existing rank-bias CSVs are reused by default in ``--run-rank-bias`` mode. Pass
``--no-reuse-rank-bias-csvs`` when changing randomness controls such as
``--no-vary-rht-sign``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .analyze_rank_bias import (
    BACKENDS,
    RECIPE_ALIASES,
    RECIPES,
    VARIANTS,
    assert_stochastic,
    bucket_names,
    export_csv,
    flatten_to_2d,
    make_rank_labels,
    plot_results,
    rht_matrices,
    sweep_recipe,
)
from .dsv3_dumps import classify as classify_dsv3
from .dsv3_dumps import discover_dsv3_tensors, load_dump_tensor

DEFAULT_RECIPES = ["9004"]
DEFAULT_RECIPE_LABELS = {
    "9004": "RHT-16 (wgrad), cast to FP4 with SR",
    "6302": "Cast to FP4 with SR",
    "nvfp4": "RNE baseline, no SR",
}
# Row-group sort priority; unrecognized prefixes sort after these.
ROW_GROUP_ORDER = ["dense", "attn", "moe/shared", "moe/routed"]
SUMMARY_FIELDNAMES = [
    "tensor",
    "recipe_id",
    "bucket",
    "bucket_label",
    "n_elements",
    "tmin",
    "tmax",
    "slope",
    "floor_ratio",
    "mse_tmin",
    "mse_tmax",
    "flat",
]


def parse_recipe_labels(items: Optional[List[str]]) -> Dict[str, str]:
    labels = dict(DEFAULT_RECIPE_LABELS)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--recipe-label must look like RECIPE=LABEL, got {item!r}")
        key, label = item.split("=", 1)
        labels[key] = label
    return labels


_VARIANT_SUFFIXES = ("_GT", "_XT", "_WT", "_G", "_X", "_W")


def classify(tensor: str) -> str:
    """Undo tensor_summary_name() (layer/expert/variant tagging), then classify."""
    stem = tensor
    for suffix in _VARIANT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    m = re.match(r"layer(\d+)_(.*)$", stem)
    if not m:
        return "unknown"
    module_name = re.sub(r"_expert\d+$", "", m.group(2))
    return classify_dsv3(module_name)


def _row_sort_key(block_type: str) -> Tuple[int, str]:
    for i, prefix in enumerate(ROW_GROUP_ORDER):
        if block_type == prefix or block_type.startswith(prefix + "/"):
            return (i, block_type)
    return (len(ROW_GROUP_ORDER), block_type)


def discover_row_types(slope_csv: str) -> List[Tuple[str, str]]:
    """Row (block_type, label) pairs actually present in a flatness summary CSV."""
    block_types = set()
    with open(slope_csv, newline="") as f:
        for row in csv.DictReader(f):
            block_types.add(classify(row["tensor"]))
    block_types.discard("unknown")
    return [(bt, bt) for bt in sorted(block_types, key=_row_sort_key)]


def load_metric(
    path: str,
    value_key: str,
    recipes: Sequence[str],
    rows: Sequence[Tuple[str, str]],
) -> Tuple[Dict[Tuple[str, str, int], List[float]], Dict[str, set]]:
    out: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    n_by: Dict[str, set] = defaultdict(set)
    wanted = {k for k, _ in rows}
    first_recipe = recipes[0] if recipes else None
    recipe_set = set(recipes)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            recipe = row["recipe_id"]
            if recipe not in recipe_set:
                continue
            bucket = int(row["bucket"])
            btype = classify(row["tensor"])
            if btype not in wanted:
                continue
            out[(btype, recipe, bucket)].append(float(row[value_key]))
            if recipe == first_recipe:
                n_by[btype].add(row["tensor"])
    return out, n_by


def cell_text(v: float) -> str:
    """Slope as |slope| x 100, zero-padded to two digits: -0.39 -> "39",
    -1.00 -> "100", -0.07 -> "07".

    Every slope here lies in [-1, 0] up to fitting noise, so the sign and the
    leading "0." are four characters of pure constant in every one of the 208
    cells. At 16 columns that is what runs adjacent labels together and makes the
    grid unreadable; the magnitude alone fits. The colorbar carries the sign.

    Padding to two digits matters next to a neighbour reading "95": a bare "7"
    invites reading 0.7 rather than 0.07, and "07" does not.

    A positive slope would be fitting noise on an all-but-flat bucket, but it is
    signed explicitly rather than silently shown as its magnitude.
    """
    scaled = abs(v) * 100.0
    return ("+" if v > 0 else "") + f"{scaled:02.0f}"


def plot_row(
    axes,
    data: Dict[Tuple[str, str, int], List[float]],
    n_by: Dict[str, set],
    *,
    recipes: Sequence[str],
    recipe_labels: Dict[str, str],
    ranks: Sequence[int],
    vmin: float,
    vmax: float,
    cmap,
    fmt,
    center_white: Optional[float] = None,
    rows: Sequence[Tuple[str, str]],
):
    row_labels = [f"{lab} (n={len(n_by[bt])})" for bt, lab in rows]
    col_labels = ["amax" if r == 1 else f"r{r}" for r in ranks]
    im = None
    for ax, recipe in zip(axes, recipes):
        mat = np.full((len(rows), len(ranks)), np.nan)
        for i, (bt, _) in enumerate(rows):
            for j, rank in enumerate(ranks):
                values = data[(bt, recipe, rank)]
                mat[i, j] = float(np.median(values)) if values else float("nan")
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ranks)))
        # Rotated: "amax" is wider than its column, so at 16 ranks the horizontal
        # labels collide ("amaxr2", "r10r11r12") and the axis is unreadable.
        ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(row_labels)
        ax.set_title(
            f"{recipe_labels.get(recipe, f'recipe {recipe}')}\n(recipe {recipe})",
            fontsize=10,
        )
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                if center_white is None:
                    text_color = "white" if v > (vmin + vmax) / 2 else "black"
                else:
                    text_color = "white" if abs(v - center_white) > 0.2 else "black"
                ax.text(
                    j,
                    i,
                    fmt(v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
    axes[0].set_ylabel("block type")
    return im


def make_summary_heatmap(
    slope_csv: str,
    out_dir: str,
    *,
    recipes: Sequence[str],
    recipe_labels: Dict[str, str],
    tag: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    rows = discover_row_types(slope_csv)
    ranks = list(range(1, 17))
    height = 4.8 + 0.35 * max(0, len(rows) - 6)

    slope_data, n_slope = load_metric(slope_csv, "slope", recipes, rows)
    # 0.45in per rank column: the widest cell label is 3 characters ("100"), and
    # at the previous fixed 4.85in the 16 columns were narrower than their own
    # text, so a row of ideal slopes rendered as "100100100100".
    panel_w = max(4.85, 0.45 * len(ranks))
    fig, axes = plt.subplots(
        1, len(recipes), figsize=(panel_w * len(recipes), height), sharey=True
    )
    axes = np.atleast_1d(axes)
    im = plot_row(
        axes,
        slope_data,
        n_slope,
        recipes=recipes,
        recipe_labels=recipe_labels,
        ranks=ranks,
        vmin=-1.05,
        vmax=0.05,
        cmap=plt.get_cmap("magma_r"),
        fmt=cell_text,
        rows=rows,
    )
    for ax in axes:
        ax.set_xlabel("element ranked by magnitude")
    fig.suptitle(
        "Rounding bias by element magnitude, darker is more biased",
        y=1.08,
        fontsize=12,
    )
    fig.text(
        0.5,
        1.02,
        "How close the trial-mean MSE is to decaying as 1/T "
        "(ideal -1; 0 = bias that won't average out). "
        "Cells are |slope| x 100: 39 is -0.39, 100 is -1.00.",
        ha="center",
        style="italic",
        fontsize=10,
        transform=fig.transFigure,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(
        r"median $s = d\,\log \mathrm{MSE}\,/\,d\,\log T$"
        "\n"
        r"$-1$: zero-mean noise ($\mathrm{MSE}\propto 1/T$)   "
        r"$0$: bias floor",
        fontsize=8,
    )
    out_path = os.path.join(out_dir, f"mse_decay_slope_heatmaps{suffix}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def tensor_summary_name(info, variant: str) -> str:
    name = f"layer{info.layer_num}_{info.module_name}"
    if info.expert_num is not None:
        name += f"_expert{info.expert_num}"
    return f"{name}_{variant.replace('.', '')}"


def fit_slope(trials: Iterable[int], mses: Iterable[float]) -> float:
    xs = []
    ys = []
    for trial, mse in zip(trials, mses):
        if trial > 0 and math.isfinite(mse) and mse > 0:
            xs.append(math.log(float(trial)))
            ys.append(math.log(float(mse)))
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(np.array(xs), np.array(ys), 1)[0])


def summarize_rank_results(
    tensor_name: str,
    results: Dict[str, Dict[int, Dict[int, float]]],
    checkpoints: List[int],
    occupied: List[int],
    names: Dict[int, str],
    counts: Any,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tmin = checkpoints[0]
    tmax = checkpoints[-1]
    for recipe_id, per_trial in results.items():
        for bucket in occupied:
            mses = [float(per_trial[t][bucket]) for t in checkpoints]
            mse_tmin = mses[0]
            mse_tmax = mses[-1]
            floor_ratio = mse_tmax / mse_tmin if mse_tmin > 0 else float("nan")
            slope = fit_slope(checkpoints, mses)
            rows.append(
                {
                    "tensor": tensor_name,
                    "recipe_id": recipe_id,
                    "bucket": bucket,
                    "bucket_label": names[bucket],
                    "n_elements": int(counts[bucket].item()),
                    "tmin": tmin,
                    "tmax": tmax,
                    "slope": slope,
                    "floor_ratio": floor_ratio,
                    "mse_tmin": mse_tmin,
                    "mse_tmax": mse_tmax,
                    "flat": int(math.isfinite(slope) and slope > -0.5),
                }
            )
    return rows


def upsert_summary_rows(path: str, new_rows: List[Dict[str, object]]) -> None:
    keyed: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["tensor"], row["recipe_id"], int(row["bucket"]))
                keyed[key] = row
    for row in new_rows:
        key = (str(row["tensor"]), str(row["recipe_id"]), int(row["bucket"]))
        keyed[key] = row

    def sort_key(item):
        tensor, recipe_id, bucket = item[0]
        m = re.match(r"layer(\d+)_(.*)$", tensor)
        layer = int(m.group(1)) if m else 10**9
        return (layer, tensor, recipe_id, bucket)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for _, row in sorted(keyed.items(), key=sort_key):
            writer.writerow({name: row.get(name, "") for name in SUMMARY_FIELDNAMES})
    print(f"Updated: {path} ({len(new_rows)} rows upserted)")


def load_reusable_rank_bias_results(
    rank_bias_dir: str,
    tensor_name: str,
    block_size: int,
    recipes: Sequence[str],
    checkpoints: Sequence[int],
    occupied: Sequence[int],
    names: Dict[int, str],
    counts: Any,
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """Load complete per-recipe results from existing rank-bias CSVs.

    The CSV schema does not record randomness controls, so callers should opt
    out of reuse when changing flags such as --no-vary-rht-sign.
    """
    if not os.path.isdir(rank_bias_dir):
        return {}

    prefix = f"rank_bias_{tensor_name}_b{block_size}_"
    partial: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    recipe_set = set(recipes)
    checkpoint_set = set(checkpoints)
    occupied_set = set(occupied)

    for filename in os.listdir(rank_bias_dir):
        if not filename.startswith(prefix) or not filename.endswith(".csv"):
            continue
        path = os.path.join(rank_bias_dir, filename)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                recipe_id = row["recipe_id"]
                if recipe_id not in recipe_set:
                    continue
                trial = int(row["trials"])
                bucket = int(row["bucket"])
                if trial not in checkpoint_set or bucket not in occupied_set:
                    continue
                if row.get("bucket_label") and row["bucket_label"] != names[bucket]:
                    continue
                if int(row.get("n_elements", counts[bucket].item())) != int(
                    counts[bucket].item()
                ):
                    continue
                partial[recipe_id][trial][bucket] = float(row["mse_of_trial_mean"])

    reusable: Dict[str, Dict[int, Dict[int, float]]] = {}
    for recipe_id in recipes:
        per_trial = partial.get(recipe_id)
        if not per_trial:
            continue
        complete = all(
            trial in per_trial and all(bucket in per_trial[trial] for bucket in occupied)
            for trial in checkpoints
        )
        if complete:
            reusable[recipe_id] = {
                int(trial): {
                    int(bucket): per_trial[trial][bucket] for bucket in occupied
                }
                for trial in checkpoints
            }
    return reusable


def run_rank_bias_batch(args) -> str:
    tensor_type, transpose = VARIANTS[args.variant]
    backend = BACKENDS[args.backend]
    recipe_ids = [RECIPE_ALIASES.get(r, r) for r in args.recipes]
    for recipe_id in recipe_ids:
        reason = RECIPES[recipe_id].unsupported.get(args.variant)
        if reason:
            raise SystemExit(f"--variant {args.variant} with --recipes {recipe_id}: {reason}")

    infos = discover_dsv3_tensors(
        args.base_dir,
        tensor_type,
        rank=args.rank,
        step=args.step,
        layer_names=args.layer_types,
        skip_layer_numbers=args.skip_layer_numbers,
        exclude_experts=args.exclude_experts,
    )
    checkpoints = sorted(set(args.trials))
    rank_bias_dir = args.rank_bias_dir or os.path.join(args.out_dir, "rank_bias")
    os.makedirs(rank_bias_dir, exist_ok=True)
    summary_csv = args.summary_csv or os.path.join(args.out_dir, "_flatness_summary.csv")
    print(f"Discovered {len(infos)} tensors in {args.base_dir}")

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

    all_summary_rows: List[Dict[str, object]] = []
    for i, info in enumerate(infos, 1):
        tensor_name = tensor_summary_name(info, args.variant)
        print(f"[{i}/{len(infos)}] {tensor_name}")
        tensor_cpu = flatten_to_2d(load_dump_tensor(info.filepath))
        labels_cpu = make_rank_labels(tensor_cpu, args.block_size, transpose=transpose)
        counts_cpu = torch.bincount(
            labels_cpu.reshape(-1).long(), minlength=args.block_size + 1
        )
        occupied = [
            bucket
            for bucket in range(1, args.block_size + 1)
            if counts_cpu[bucket] > 0
        ]
        if counts_cpu[0] > 0:
            occupied.append(0)
        names = bucket_names(occupied)

        results = (
            {}
            if args.no_reuse_rank_bias_csvs
            else load_reusable_rank_bias_results(
                rank_bias_dir,
                tensor_name,
                args.block_size,
                recipe_ids,
                checkpoints,
                occupied,
                names,
                counts_cpu,
            )
        )
        missing_recipes = [r for r in recipe_ids if r not in results]
        if results:
            reused = " ".join(results)
            print(f"  reusing existing CSV results for recipes: {reused}")

        tensor = labels = counts = None
        if missing_recipes:
            tensor = tensor_cpu.cuda()
            labels = labels_cpu.cuda()
            counts = counts_cpu.cuda().double()
        for recipe_id in missing_recipes:
            use_sr, use_rht = settings[recipe_id]
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
            )
            torch.cuda.empty_cache()

        results = {recipe_id: results[recipe_id] for recipe_id in recipe_ids}
        stem = f"rank_bias_{tensor_name}_b{args.block_size}_{'_'.join(recipe_ids)}"
        export_csv(
            os.path.join(rank_bias_dir, stem + ".csv"),
            results,
            checkpoints,
            occupied,
            names,
            counts_cpu,
        )
        if not args.no_rank_bias_plots:
            plot_results(
                os.path.join(rank_bias_dir, stem + ".png"),
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
        all_summary_rows.extend(
            summarize_rank_results(
                tensor_name, results, checkpoints, occupied, names, counts_cpu
            )
        )
        del tensor, labels, counts, tensor_cpu, labels_cpu, counts_cpu
        torch.cuda.empty_cache()

    upsert_summary_rows(summary_csv, all_summary_rows)
    report_sweep_metrics(all_summary_rows, recipe_ids)
    return summary_csv


def report_sweep_metrics(
    rows: Sequence[Dict[str, object]], recipe_ids: Sequence[str]
) -> None:
    """Print flat, greppable scalars for a CI metric scraper.

    A batch sweep produces one CSV row per (tensor, recipe, bucket), which no
    stdout_regex metric can reduce. These four lines are the reduction: how many
    tensors were swept, and -- over the AMAX bucket only, which is the one every
    single-tensor run has found to be the offender -- the worst and median slope
    and how many tensors the summariser marked flat.

    The recipe id is in the metric name because --recipes can sweep several in
    one run and a stdout_regex metric cannot associate two lines with each other.
    """
    for recipe_id in recipe_ids:
        amax = [
            r
            for r in rows
            if str(r.get("recipe_id")) == str(recipe_id) and int(r.get("bucket", -1)) == 1
        ]
        slopes = sorted(
            float(r["slope"])
            for r in amax
            if r.get("slope") not in ("", None) and not math.isnan(float(r["slope"]))
        )
        n_flat = sum(1 for r in amax if str(r.get("flat")).lower() in ("true", "1"))
        print(f"heatmap_{recipe_id}_tensors_swept: {len(amax)}")
        print(f"heatmap_{recipe_id}_amax_flat_count: {n_flat}")
        if not slopes:
            continue
        mid = len(slopes) // 2
        median = slopes[mid] if len(slopes) % 2 else (slopes[mid - 1] + slopes[mid]) / 2
        # Worst = least negative = flattest = the bucket averaging does not help.
        print(f"heatmap_{recipe_id}_amax_worst_slope: {slopes[-1]:.6f}")
        print(f"heatmap_{recipe_id}_amax_median_slope: {median:.6f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slope-csv", default=None, help="Existing flatness summary CSV.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument(
        "--recipes",
        type=str,
        nargs="+",
        default=None,
        help="Recipe IDs to analyze and plot. Required with --run-rank-bias; "
        "defaults to 9004 in plot-only mode.",
    )
    p.add_argument(
        "--recipe-label",
        action="append",
        default=None,
        help="Repeatable RECIPE=LABEL override for heatmap panel titles.",
    )

    p.add_argument(
        "--run-rank-bias",
        action="store_true",
        help="Run analyze_rank_bias over discovered DSV3 tensors before plotting.",
    )
    p.add_argument("--base-dir", default=None, help="DSV3 dump directory.")
    p.add_argument(
        "--variant",
        type=str,
        default="G",
        choices=list(VARIANTS),
        help="Leaf QDQ path: X, X.T, W, W.T, G, G.T.",
    )
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--layer-types", nargs="+", default=None)
    p.add_argument("--skip-layer-numbers", type=int, nargs="+", default=None)
    p.add_argument("--exclude-experts", action="store_true")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--trials", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="cutedsl",
        help="QDQ implementation (both are bitwise identical; default: cutedsl).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base Philox seed.")
    p.add_argument("--rank-bias-dir", default=None)
    p.add_argument("--summary-csv", default=None)
    p.add_argument(
        "--no-reuse-rank-bias-csvs",
        action="store_true",
        help="Recompute all recipes even when matching rank-bias CSVs exist. "
        "Use this when changing randomness controls.",
    )
    p.add_argument("--no-rank-bias-plots", action="store_true")
    p.add_argument(
        "--vary-rht-sign",
        dest="vary_rht_sign",
        action="store_true",
        default=False,
        help="Re-draw the RHT sign vector every trial (see analyze_rank_bias.py).",
    )
    p.add_argument(
        "--no-vary-rht-sign", dest="vary_rht_sign", action="store_false"
    )
    p.add_argument("--skip-stochastic-check", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.run_rank_bias:
        if not args.base_dir:
            raise SystemExit("--base-dir is required with --run-rank-bias")
        if not args.recipes:
            raise SystemExit("--recipes is required with --run-rank-bias")
        if len(set(args.trials)) < 2 or min(args.trials) < 1:
            raise SystemExit("--trials needs at least two distinct positive checkpoints")
        if args.replicas < 1:
            raise SystemExit("--replicas must be >= 1")
        if args.block_size < 1:
            raise SystemExit("--block-size must be >= 1")
        slope_csv = run_rank_bias_batch(args)
    else:
        if not args.slope_csv:
            raise SystemExit("--slope-csv is required unless --run-rank-bias is set")
        slope_csv = args.slope_csv
        if not args.recipes:
            args.recipes = DEFAULT_RECIPES

    recipe_labels = parse_recipe_labels(args.recipe_label)
    # A recipe's label is not lane-aware, but 9004's RHT is: it rotates only the
    # TRANSPOSE lane of X and G, never an identity lane and never W. Titling an
    # identity-lane heatmap "RHT-16 (wgrad)" claims a transform that was not
    # applied, which is exactly the class of error this whole analysis has
    # already been burned by once. Strike it when the lane under test is not
    # rotated.
    _, _transpose = VARIANTS[args.variant]
    for _rid in list(recipe_labels):
        _base = RECIPE_ALIASES.get(_rid, _rid)
        _rotates = _base in RECIPES and bool(RECIPES[_base].rht_transpose)
        if _rotates and not _transpose:
            recipe_labels[_rid] = recipe_labels[_rid].replace(
                "RHT-16 (wgrad), ", ""
            ) + " (identity lane: no RHT)"
    make_summary_heatmap(
        slope_csv,
        args.out_dir,
        recipes=args.recipes,
        recipe_labels=recipe_labels,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
