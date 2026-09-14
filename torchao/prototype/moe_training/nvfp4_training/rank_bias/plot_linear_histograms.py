"""Signed value histograms on LINEAR axes, before and after NVFP4, FC1 over FC2.

A different figure from plot_activation_histograms, not a flag on it. That one
bins log10(|x| / block_amax) with log y, which is the right coordinate for the
quantizer's own question -- where the FP4 code points and the 1/24 flush
threshold sit -- but it throws away the sign and normalizes each element by its
own block, so it cannot show the SHAPE of a distribution. This one keeps the
raw signed value and linear axes, which is how a distribution is normally read.

LAYOUT, and it is the point of the figure:

    row 1   FC1 input   |  dense raw | dense NVFP4 | shared raw | shared NVFP4 | ...
    row 2   FC2 input   |  ...

Before and after are ADJACENT columns per family, and FC1 sits directly above
FC2, so the gate's effect reads down a column pair and the quantizer's effect
reads across one.

Per-tensor, not pooled: raw value ranges differ by orders of magnitude between
families (amax 0.498 on dense fc2, 258 on layer60 shared), so a pooled linear
histogram would be one spike and five empty axes. Each panel is one named
tensor, binned over its own range, exactly as the Nemotron reference figure does.
"""
import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from .analyze_rank_bias import (
    BACKENDS,
    RECIPE_ALIASES,
    RECIPES,
    VARIANTS,
    flatten_to_2d,
)
from .dsv3_dumps import classify, discover_dsv3_tensors, load_dump_tensor
from .analyze_sparsity import _slug
from .plot_activation_histograms import _quant_dequant, qsnr_db


def excess_kurtosis(x: torch.Tensor) -> float:
    x = x.float().flatten()
    mu, sd = x.mean(), x.std()
    if not torch.isfinite(sd) or sd == 0:
        return float("nan")
    return float(((x - mu) / sd).pow(4).mean() - 3.0)


def _hist_linear(x: torch.Tensor, bins: int, lo: float, hi: float) -> torch.Tensor:
    return torch.histc(x.float().flatten(), bins=bins, min=lo, max=hi).cpu()


def _panel(ax, centers, frac, color, label, lines) -> None:
    ax.fill_between(centers, frac, 0.0, color=color, alpha=0.22)
    ax.plot(centers, frac, color=color, linewidth=1.5)
    ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
    ax.text(
        0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=7,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.5),
    )
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tensor-type", default="X", choices=["X", "W", "G"])
    p.add_argument("--variant", default="X", choices=sorted(VARIANTS))
    p.add_argument("--recipe", default="9004", choices=sorted(set(RECIPES) | set(RECIPE_ALIASES)))
    p.add_argument("--backend", choices=sorted(BACKENDS), default="cutedsl")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--tag", default=None)
    p.add_argument("--csv", default=None, help="Write the linear bin counts here.")
    # One tensor per column pair. FC1 row first, then FC2, same families in the
    # same order -- the script does not guess the pairing, the caller states it.
    p.add_argument("--fc1", nargs="+", required=True, help="FC1 tensor names, left to right.")
    p.add_argument("--fc2", nargs="+", required=True, help="FC2 tensor names, same families, same order.")
    p.add_argument("--labels", nargs="+", default=None, help="Column labels, one per pair.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    recipe_id = RECIPE_ALIASES.get(args.recipe, args.recipe)
    reason = RECIPES[recipe_id].unsupported.get(args.variant)
    if reason:
        raise SystemExit(f"--variant {args.variant} with --recipe {recipe_id}: {reason}")
    if len(args.fc1) != len(args.fc2):
        raise SystemExit(f"--fc1 has {len(args.fc1)} names, --fc2 has {len(args.fc2)}; must pair")
    backends = BACKENDS[args.backend]
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    wanted = list(args.fc1) + list(args.fc2)
    infos = discover_dsv3_tensors(
        args.base_dir, args.tensor_type, rank=args.rank, step=args.step
    )

    def tensor_name(info) -> str:
        name = f"layer{info.layer_num}_{info.module_name}"
        return name if info.expert_num is None else f"{name}_expert{info.expert_num}"

    by_name = {tensor_name(i): i for i in infos}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(
            f"not in {args.base_dir}: {missing}\n"
            f"(discovered {len(infos)} tensors; check --rank/--step and the dump arm)"
        )

    # One quantize pass per requested tensor, then bin both on the RAW tensor's
    # range so the two panels of a pair share an x axis and are comparable.
    panels: Dict[str, Dict[str, object]] = {}
    rows_csv: List[Tuple[str, str, int, float, float, float]] = []
    for name in wanted:
        info = by_name[name]
        raw = flatten_to_2d(load_dump_tensor(info.filepath)).to(device)
        if raw.numel() == 0:
            raise SystemExit(f"{name}: 0 elements (expert received no tokens); pick another")
        dq = _quant_dequant(
            raw, recipe_id=recipe_id, variant=args.variant,
            tensor_type=args.tensor_type, backends=backends, seed=args.seed,
        )
        lo = float(raw.abs().max())
        edges = torch.linspace(-lo, lo, args.bins + 1)
        centers = (0.5 * (edges[:-1] + edges[1:])).tolist()
        h_raw = _hist_linear(raw, args.bins, -lo, lo)
        h_dq = _hist_linear(dq, args.bins, -lo, lo)
        panels[name] = {
            "centers": centers,
            "raw": (h_raw / h_raw.sum()).tolist(),
            "dq": (h_dq / h_dq.sum()).tolist(),
            "shape": tuple(raw.shape),
            "amax": lo,
            "kurt_raw": excess_kurtosis(raw),
            "kurt_dq": excess_kurtosis(dq),
            "qsnr": qsnr_db(raw, dq),
            "family": classify(info.module_name),
        }
        for kind in ("raw", "dq"):
            for i, c in enumerate(centers):
                rows_csv.append((name, kind, i, c, panels[name][kind][i], lo))
        print(f"binned {name}: {tuple(raw.shape)} amax {lo:.4g} qsnr {panels[name]['qsnr']:.2f} dB")
        del raw, dq
        torch.cuda.empty_cache()

    npair = len(args.fc1)
    # Column labels name the FAMILY, not the row. Deriving them from args.fc1
    # and reusing them on the FC2 row titled every bottom panel "...fc1" while
    # the data under it was fc2 -- the figure lied about itself. The family
    # suffix is stripped for the same reason: "moe/shared" is the column,
    # fc1-vs-fc2 is the row, and the exact tensor is named in the panel box.
    labels = args.labels if args.labels and len(args.labels) == npair else [
        str(panels[n]["family"]).rsplit("/", 1)[0] for n in args.fc1
    ]
    fig, axes = plt.subplots(2, 2 * npair, figsize=(3.05 * 2 * npair, 6.4), squeeze=False)
    for ri, (row_names, row_title) in enumerate(
        ((args.fc1, "FC1 input"), (args.fc2, "FC2 input"))
    ):
        for ci, name in enumerate(row_names):
            p = panels[name]
            for k, (kind, color, what) in enumerate(
                (("raw", "tab:red", "raw"), ("dq", "tab:blue", f"after NVFP4 {recipe_id}"))
            ):
                ax = axes[ri][2 * ci + k]
                lines = [
                    name,
                    f"shape {p['shape'][0]}x{p['shape'][1]}",
                    f"excess kurtosis {p['kurt_raw' if kind == 'raw' else 'kurt_dq']:.3g}",
                ]
                if kind == "dq":
                    lines.append(f"QSNR {p['qsnr']:.2f} dB")
                _panel(ax, p["centers"], p[kind], color, f"{labels[ci]}\n{what}", lines)
                ax.set_xlabel(f"{args.tensor_type} value", fontsize=7.5)
                if 2 * ci + k == 0:
                    ax.set_ylabel(f"{row_title}\nfraction per bin", fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.text(
        0.5, 0.975,
        f"{args.tensor_type} value histograms before and after NVFP4 — FC1 over FC2",
        ha="center", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.935,
        f"recipe {recipe_id} · variant {args.variant} · rank {args.rank} · step {args.step} · "
        f"{args.bins} linear bins over each tensor's own +/-amax · linear y · "
        "raw and quantized share an x axis within a pair",
        ha="center", fontsize=8, color="0.35",
    )
    out = os.path.join(args.out_dir, f"linear_histograms{suffix}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tensor", "kind", "bin", "bin_center", "fraction", "amax"])
            for r in rows_csv:
                w.writerow([r[0], r[1], r[2], f"{r[3]:.6g}", f"{r[4]:.8e}", f"{r[5]:.6g}"])
        print(f"wrote {args.csv}")

    stem = f"linhist_{_slug(recipe_id)}"
    print()
    print(f"{stem}_panels: {len(panels)}")
    for name, p in panels.items():
        print(f"{stem}_{_slug(name)}_qsnr_db: {p['qsnr']:.6f}")
        print(f"{stem}_{_slug(name)}_excess_kurtosis_raw: {p['kurt_raw']:.6f}")


if __name__ == "__main__":
    main()
