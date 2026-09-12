"""Value histograms of an activation dump, before and after NVFP4.

WHAT THIS ANSWERS. The FC2 input of a SwiGLU FFN is ``silu(w1(x)) * w3(x)`` --
the output of the gated activation, and the operand whose shape decides how much
of the expert down-projection NVFP4 can represent. A squared-ReLU model
(Nemotron) produces a far more extreme near-zero mass there than DSV3's SwiGLU
does, so the distribution itself, not just its reductions, is the thing worth
looking at. This draws it twice per panel: as dumped, and as the quantizer hands
it back.

WHAT IT DELIBERATELY DOES NOT DO. It computes no sparsity statistic of its own.
``analyze_sparsity.py`` already computes every one the question names -- flush
(kitchen ``Metric.FTZ``), ``block_nnz_p50``/``p05`` per 1x16 block,
``dead_block`` (``BLOCK_SCALE_FTZ``), ``ftz_thresh`` (``FTZ_THRESHOLD``) -- and
this reads its CSV via ``--sparsity-csv`` to annotate the panels. Two scripts,
one quantization pass each, no second definition of the same number.

THE X AXIS IS ``log10(|x| / block_amax)``, NOT THE VALUE. A signed linear axis
cannot draw this distribution: excess kurtosis runs from 91 (routed experts) to
1.8e4 (dense), so over ``[-amax, +amax]`` every element lands in the one or two
bins astride zero and the quantized row is visually identical to the raw one --
NVFP4 moves each element by a fraction of its own block amax, far below one bin
width. Block-relative magnitude is also the quantity the format actually acts
on: an element rounds to FP4 zero at ``|x| / block_amax < 1/24``, drawn as a
dashed line on every panel. Flushed elements have ``|x| == 0`` and pile into the
leftmost bin, so FTZ reads off the figure as mass rather than out of a caption.
A log y axis then keeps the tail -- the part the format destroys -- visible
against a peak three decades above it.

The axis is therefore FIXED at ``[-6, 0]`` for every panel, which makes all
tensors and both rows directly comparable and needs no range-finding pass over
the dump.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from .analyze_rank_bias import (
    BACKENDS,
    GEMM_TYPES,
    RECIPE_ALIASES,
    RECIPES,
    VARIANTS,
    flatten_to_2d,
)
from .analyze_sparsity import FP4_ZERO_RATIO, _slug, _sr_kwarg
from .dsv3_dumps import classify, discover_dsv3_tensors, load_dump_tensor
from .nvfp4_reference import BLOCK_SIZE

# Fixed x-range, in decades of |x| / block_amax. 0 is the block amax itself; -6
# collects everything below a millionth of it, including every flushed element.
LOG_MIN, LOG_MAX = -6.0, 0.0

# Row order of the pooled figure's columns. Anything classify() returns that is
# not listed is appended after these, so a new family shows up rather than
# vanishing.
FAMILY_ORDER = ("dense/fc2", "moe/shared/fc2", "moe/routed/fc2")


def qsnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Quantization SNR in dB, kitchen's definition (metrics_utils.py:254).

    ``20 * log10(||ref||_F / ||ref - approx||_F)``, in fp32. Ported rather than
    imported: kitchen is not on this image's path.
    """
    ref = ref.float()
    err = (ref - approx.float()).norm()
    if err == 0:
        return float("inf")
    return float(20.0 * torch.log10(ref.norm() / err))


def excess_kurtosis(x: torch.Tensor) -> float:
    """``E[(x-mu)^4] / sigma^4 - 3``; 0 for a Gaussian, positive for heavy tails."""
    x = x.float().flatten()
    centered = x - x.mean()
    var = centered.pow(2).mean()
    if var == 0:
        return float("nan")
    return float(centered.pow(4).mean() / var.pow(2) - 3.0)


def _quant_dequant(tensor: torch.Tensor, *, recipe_id: str, variant: str,
                   tensor_type: str, backends, seed: int) -> torch.Tensor:
    """Round-trip ``tensor`` through the recipe's quantizer for this lane.

    The X fprop lane is unrotated under every recipe that accepts X, so the
    returned tensor is element-aligned with the input and a value histogram of
    the two is a like-for-like comparison. ``matrices``-style rotation handling
    is therefore deliberately absent; ``analyze_one`` is where the rotated lanes
    live.
    """
    recipe = RECIPES[recipe_id]
    _, transpose = VARIANTS[variant]
    gemm = GEMM_TYPES[(tensor_type, transpose)]
    if gemm in recipe.rht_gemms:
        raise ValueError(
            f"recipe {recipe_id} rotates the {gemm} lane, so a value histogram of "
            f"variant {variant} would compare two different bases. Use a lane the "
            f"recipe leaves unrotated."
        )
    backend = backends[recipe.quantizer[tensor_type]]
    return backend.quant_dequant(
        tensor, transpose=transpose, seed=seed, **{_sr_kwarg(backend): False}
    )


def block_amax(x: torch.Tensor) -> torch.Tensor:
    """Per-1x16-block amax of a 2D tensor, in the blocked layout ``_blocks`` returns."""
    return _blocks(x).abs().amax(dim=-1, keepdim=True)


def _blocks(x: torch.Tensor) -> torch.Tensor:
    """``[rows, cols] -> [rows, cols // 16, 16]``, the NVFP4 scaling layout along K."""
    rows, cols = x.shape
    if cols % BLOCK_SIZE:
        raise SystemExit(
            f"last dim {cols} is not a multiple of the NVFP4 block size {BLOCK_SIZE}; "
            "this figure assumes the unrotated fprop layout, where it always is"
        )
    return x.float().reshape(rows, cols // BLOCK_SIZE, BLOCK_SIZE)


def relative_log(x: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """``log10(|x| / amax)`` per element, clamped into ``[LOG_MIN, LOG_MAX]``.

    ``x`` is blocked; ``amax`` is the RAW tensor's per-block amax, used for both
    rows so the two are on one scale. An element the quantizer flushed is exactly
    0, so its log is -inf and it clamps into the leftmost bin -- that pile-up is
    the FTZ mass. An all-zero block has no amax to divide by and contributes there
    too.
    """
    rel = torch.where(amax > 0, x.abs() / amax, torch.zeros_like(x))
    return rel.log10().clamp_(LOG_MIN, LOG_MAX).flatten()


def _hist(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Counts of ``x`` over ``bins`` equal bins spanning ``[LOG_MIN, LOG_MAX]``."""
    return torch.histc(x.float(), bins=bins, min=LOG_MIN, max=LOG_MAX).cpu()


def _bin_centers(bins: int) -> torch.Tensor:
    edges = torch.linspace(LOG_MIN, LOG_MAX, bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _draw(ax, counts: torch.Tensor, color: str) -> None:
    total = float(counts.sum())
    frac = counts / total if total else counts
    centers = _bin_centers(len(counts))
    ax.fill_between(centers, frac, 1e-9, color=color, alpha=0.25)
    ax.plot(centers, frac, color=color, linewidth=1.4)
    ax.axvline(math.log10(FP4_ZERO_RATIO), color="0.25", linestyle="--", linewidth=0.9)
    # A hair wider than the data, so the two edge bins -- the flushed pile-up at
    # LOG_MIN and the block amax itself at 0 -- do not render on the spines.
    ax.set_xlim(LOG_MIN - 0.15, LOG_MAX + 0.15)
    ax.set_yscale("log")
    # Log y, so headroom for the top-left annotation box is decades, not a
    # fraction. The floor is fixed: a bin holding one element in 1e7 is noise.
    peak = float(frac.max()) if len(frac) else 0.0
    if peak > 0:
        ax.set_ylim(1e-7, peak * 60)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)


def _annotate(ax, lines: Sequence[str]) -> None:
    # Bottom-left, not top-left: on a log y axis the top-left corner is where the
    # flushed mass piles up, which is the one thing the box must not cover.
    ax.text(
        0.02, 0.03, "\n".join(lines), transform=ax.transAxes,
        va="bottom", ha="left", fontsize=6.5,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.5),
    )


def read_sparsity_csv(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """``tensor`` -> row, from an ``analyze_sparsity.py --csv`` export.

    Empty when no CSV is given; the panels then carry distribution statistics
    only. Rows for other variants are ignored -- one lane per figure.
    """
    if not path:
        return {}
    with open(path, newline="") as f:
        return {row["tensor"]: row for row in csv.DictReader(f)}


def _median(entries: Sequence[Dict[str, object]], key: str) -> float:
    values = sorted(float(e[key]) for e in entries if math.isfinite(float(e[key])))
    return values[len(values) // 2] if values else float("nan")


def _csv_values(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    return sorted(float(r[key]) for r in rows if r.get(key) not in (None, ""))


def _median_csv(rows: Sequence[Dict[str, str]], key: str) -> Optional[float]:
    values = _csv_values(rows, key)
    return values[len(values) // 2] if values else None


def _min_csv(rows: Sequence[Dict[str, str]], key: str) -> Optional[float]:
    values = _csv_values(rows, key)
    return values[0] if values else None


def _max_csv(rows: Sequence[Dict[str, str]], key: str) -> Optional[float]:
    values = _csv_values(rows, key)
    return values[-1] if values else None


def _fmt(row: Optional[Dict[str, str]], key: str, unit: str = "%") -> Optional[str]:
    if not row or key not in row or row[key] == "":
        return None
    return f"{float(row[key]):.2f}{unit}"


def build_figure(
    columns: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor, Dict[str, object]]],
    *,
    title: str,
    subtitle: str,
    raw_label: str,
    quant_label: str,
    out_path: str,
) -> None:
    """Two rows -- raw on top, dequantized below -- one column per entry.

    ``columns`` entries are ``(title, subtitle, raw_counts, dq_counts, stats)``. Shape and conventions follow ``plot_bias_heatmaps.make_summary_heatmap``:
    Agg, ``dpi=160``, ``bbox_inches="tight"``.
    """
    n = len(columns)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 5.6), squeeze=False)
    for col, (col_title, col_sub, raw_counts, dq_counts, stats) in enumerate(columns):
        top, bottom = axes[0][col], axes[1][col]
        top.set_title(f"{col_title}\n{col_sub}", fontsize=8.5, fontweight="bold")
        _draw(top, raw_counts, "tab:red")
        _draw(bottom, dq_counts, "tab:blue")
        _annotate(top, stats["raw_lines"])
        _annotate(bottom, stats["quant_lines"])
        bottom.set_xlabel("log10( |x| / block amax )", fontsize=7.5)
        if col == 0:
            top.set_ylabel(f"{raw_label}\nfraction per bin", fontsize=8)
            bottom.set_ylabel(f"{quant_label}\nfraction per bin", fontsize=8)
    # tight_layout first, then the two header lines above the axes: placing them
    # first lets tight_layout reclaim their space and stack them on each other.
    fig.tight_layout()
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.07)
    fig.text(0.5, 1.015, subtitle, ha="center", fontsize=7.5, color="0.35")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def report_histogram_metrics(per_tensor: List[Dict[str, object]], *, recipe_id: str) -> None:
    """Flat ``name: value`` scalars for a CI metric scraper.

    Same contract and same reason as ``analyze_sparsity.report_sparsity_metrics``
    and ``plot_bias_heatmaps.report_sweep_metrics``: the script that knows what
    its columns mean is the script that defines the metric, and a hand run gets
    the same lines a CI run does.
    """
    stem = f"hist_{_slug(recipe_id)}"
    print(f"{stem}_tensors: {len(per_tensor)}")
    if not per_tensor:
        return
    by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in per_tensor:
        by_family[str(entry["family"])].append(entry)
    for family, entries in sorted(by_family.items()):
        fstem = f"{stem}_{_slug(family)}"
        finite = [float(e["qsnr_db"]) for e in entries if math.isfinite(float(e["qsnr_db"]))]
        kurt = sorted(float(e["excess_kurtosis"]) for e in entries)
        print(f"{fstem}_tensors: {len(entries)}")
        print(f"{fstem}_excess_kurtosis_median: {kurt[len(kurt) // 2]:.6f}")
        if finite:
            print(f"{fstem}_qsnr_db_median: {sorted(finite)[len(finite) // 2]:.6f}")
            print(f"{fstem}_qsnr_db_min: {min(finite):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Value histograms of a DSV3 activation dump, before and after NVFP4."
    )
    parser.add_argument("--base-dir", required=True, help="Tree of DSV3 .pt dumps.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tensor-type", default="X", choices=["X", "W", "G"])
    parser.add_argument("--variant", default="X", choices=sorted(VARIANTS))
    parser.add_argument(
        "--recipe", default="9004",
        choices=sorted(set(RECIPES) | set(RECIPE_ALIASES)),
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="cutedsl")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-types", nargs="+", default=None)
    parser.add_argument("--skip-layer-numbers", type=int, nargs="+", default=None)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--sparsity-csv", default=None,
        help="analyze_sparsity.py --csv export, used for the panel annotations.",
    )
    parser.add_argument(
        "--samples", nargs="+", default=None,
        help="Tensor names for the per-tensor figure, e.g. layer3_shared_fc2. "
             "The selection belongs on the command line, not in the code.",
    )
    parser.add_argument("--tag", default=None, help="Suffix for the output filenames.")
    parser.add_argument("--csv", default=None, help="Write the per-tensor distribution stats here.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recipe_id = RECIPE_ALIASES.get(args.recipe, args.recipe)
    reason = RECIPES[recipe_id].unsupported.get(args.variant)
    if reason:
        raise SystemExit(f"--variant {args.variant} with --recipe {recipe_id}: {reason}")
    backends = BACKENDS[args.backend]
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    infos = discover_dsv3_tensors(
        args.base_dir,
        args.tensor_type,
        rank=args.rank,
        step=args.step,
        layer_names=args.layer_types,
        skip_layer_numbers=args.skip_layer_numbers,
    )
    print(f"Discovered {len(infos)} {args.tensor_type} tensors in {args.base_dir}")

    def tensor_name(info) -> str:
        name = f"layer{info.layer_num}_{info.module_name}"
        return name if info.expert_num is None else f"{name}_expert{info.expert_num}"

    sparsity = read_sparsity_csv(args.sparsity_csv)
    pooled_raw: Dict[str, torch.Tensor] = {}
    pooled_dq: Dict[str, torch.Tensor] = {}
    pooled_n: Dict[str, int] = defaultdict(int)
    samples: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}
    per_tensor: List[Dict[str, object]] = []
    wanted = set(args.samples or ())
    empty: List[str] = []

    # One pass: quantize, then bin both rows against the raw tensor's block amax.
    for info in infos:
        name = tensor_name(info)
        family = classify(info.module_name)
        tensor = load_dump_tensor(info.filepath)
        if tensor.numel() == 0:
            # An expert that received no tokens still gets a file. Nothing to
            # bin, and the CuTe kernels fail on an empty grid with CUDA error 9
            # rather than returning -- the same guard analyze_sparsity uses.
            print(f"skipping {name}: 0 elements (expert received no tokens)")
            empty.append(name)
            continue
        tensor = flatten_to_2d(tensor).to(device)
        dq = _quant_dequant(
            tensor, recipe_id=recipe_id, variant=args.variant,
            tensor_type=args.tensor_type, backends=backends, seed=args.seed,
        )
        amax = block_amax(tensor)
        raw_counts = _hist(relative_log(_blocks(tensor), amax), args.bins)
        dq_counts = _hist(relative_log(_blocks(dq), amax), args.bins)
        if family not in pooled_raw:
            pooled_raw[family] = torch.zeros(args.bins)
            pooled_dq[family] = torch.zeros(args.bins)
        pooled_raw[family] += raw_counts
        pooled_dq[family] += dq_counts
        pooled_n[family] += 1

        entry = {
            "tensor": name,
            "family": family,
            "module": info.module_name,
            "rows": tensor.shape[0],
            "cols": tensor.shape[1],
            "amax": float(tensor.abs().max()),
            "excess_kurtosis": excess_kurtosis(tensor),
            "qsnr_db": qsnr_db(tensor, dq),
        }
        per_tensor.append(entry)

        if name in wanted:
            # Same fixed axis as the pooled figure -- these panels differ by which
            # tensor they hold, not by scale, so they can be read against it.
            samples[name] = (raw_counts, dq_counts, family)
        del tensor, dq
        torch.cuda.empty_cache()

    missing = wanted - set(samples)
    if missing:
        print(f"WARNING: --samples not found in the dump: {', '.join(sorted(missing))}")

    def stats_for(name: str, entry: Dict[str, object]) -> Dict[str, object]:
        row = sparsity.get(name)
        raw_lines = [f"{entry['rows']}x{entry['cols']}",
                     f"excess kurtosis {float(entry['excess_kurtosis']):.3g}",
                     f"amax {float(entry['amax']):.3g}"]
        below = _fmt(row, "below_1_24_pct")
        if below:
            raw_lines.append(f"|x|/block_amax < 1/24  {below}")
        quant_lines = [f"QSNR {float(entry['qsnr_db']):.2f} dB"]
        for key, label in (
            ("flush_pct", "flush"),
            ("block_nnz_p50_pct", "block_nnz p50"),
            ("block_nnz_p05_pct", "block_nnz p05"),
            ("dead_block_pct", "dead_block"),
        ):
            value = _fmt(row, key)
            if value:
                quant_lines.append(f"{label} {value}")
        return {"raw_lines": raw_lines, "quant_lines": quant_lines}

    by_name = {str(e["tensor"]): e for e in per_tensor}
    families = [f for f in FAMILY_ORDER if f in pooled_raw]
    families += [f for f in sorted(pooled_raw) if f not in FAMILY_ORDER]

    if families:
        columns = []
        for family in families:
            members = [e for e in per_tensor if e["family"] == family]
            # Medians over the family, not over the pooled element stream: a
            # per-tensor statistic summarized across tensors is the same axis
            # report_sparsity_metrics reduces on, so the two agree.
            csv_rows = [r for r in sparsity.values() if r.get("family") == family]
            stats = {
                "raw_lines": [
                    f"{len(members)} tensors, {sum(int(e['rows']) for e in members)}"
                    f"x{int(members[0]['cols'])} total",
                    f"median excess kurtosis {_median(members, 'excess_kurtosis'):.3g}",
                    f"amax {max(float(e['amax']) for e in members):.3g}",
                ],
                "quant_lines": [
                    f"median QSNR {_median(members, 'qsnr_db'):.2f} dB",
                ],
            }
            below = _median_csv(csv_rows, "below_1_24_pct")
            if below is not None:
                stats["raw_lines"].append(f"median |x|/block_amax < 1/24  {below:.2f}%")
            for key, label in (
                ("flush_pct", "median flush"),
                ("block_nnz_p50_pct", "median block_nnz p50"),
                ("block_nnz_p05_pct", "min block_nnz p05"),
                ("dead_block_pct", "max dead_block"),
            ):
                value = (
                    _min_csv(csv_rows, key) if key == "block_nnz_p05_pct"
                    else _max_csv(csv_rows, key) if key == "dead_block_pct"
                    else _median_csv(csv_rows, key)
                )
                if value is not None:
                    stats["quant_lines"].append(f"{label} {value:.2f}%")
            columns.append((
                family, f"pooled over {pooled_n[family]} tensors",
                pooled_raw[family], pooled_dq[family], stats,
            ))
        build_figure(
            columns,
            title=f"{args.tensor_type} value histograms before and after NVFP4",
            subtitle=(
                f"recipe {recipe_id} · variant {args.variant} · rank {args.rank} · "
                f"step {args.step} · {args.bins} bins over log10(|x| / block amax) · "
                f"dashed line = FP4 flush threshold 1/24 · log y · "
                "every element of every tensor included"
            ),
            raw_label="raw values",
            quant_label=f"after NVFP4 ({recipe_id})",
            out_path=os.path.join(args.out_dir, f"value_histograms_pooled{suffix}.png"),
        )

    if samples:
        ordered = [n for n in (args.samples or ()) if n in samples]
        columns = []
        for name in ordered:
            raw_counts, dq_counts, family = samples[name]
            columns.append((
                name, family, raw_counts, dq_counts,
                stats_for(name, by_name[name]),
            ))
        build_figure(
            columns,
            title=f"{args.tensor_type} value histograms before and after NVFP4 — samples",
            subtitle=(
                f"recipe {recipe_id} · variant {args.variant} · rank {args.rank} · "
                f"step {args.step} · {args.bins} bins over log10(|x| / block amax) · "
                f"dashed line = FP4 flush threshold 1/24 · log y · "
                "one named tensor per column"
            ),
            raw_label="raw values",
            quant_label=f"after NVFP4 ({recipe_id})",
            out_path=os.path.join(args.out_dir, f"value_histograms_samples{suffix}.png"),
        )

    print()
    report_histogram_metrics(per_tensor, recipe_id=recipe_id)
    print(f"hist_{_slug(recipe_id)}_tensors_empty: {len(empty)}")
    print(
        f"\nFP4 flush threshold |x| / block_amax = {FP4_ZERO_RATIO:.4f}; "
        "flush / block_nnz / dead_block annotations come from --sparsity-csv"
    )

    if args.csv:
        fields = ["tensor", "family", "module", "rows", "cols", "amax",
                  "excess_kurtosis", "qsnr_db"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_tensor)
        print(f"Exported: {args.csv}")


if __name__ == "__main__":
    main()
