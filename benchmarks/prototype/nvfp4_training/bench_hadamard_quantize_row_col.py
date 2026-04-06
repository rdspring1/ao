# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax
from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
N_SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def run_experiment(config: ExperimentConfig) -> ExperimentResult | None:
    m, n = config.m, config.n
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)

    try:
        col_amax, row_amax = triton_rht_amax(x)
        time_us = benchmark_cuda_function_in_microseconds(
            triton_rht_quantize_row_col,
            x,
            col_global_amax=col_amax,
            row_global_amax=row_amax,
        )
    except NotImplementedError:
        return None

    read_bytes = m * n * 2  # bfloat16 input
    col_write = n * (m // 2) + (n // 128) * (m // 64) * 32 * 16
    row_write = m * (n // 2) + (m // 128) * (n // 64) * 32 * 16
    total_bytes = read_bytes + col_write + row_write
    gbps = (total_bytes / 1e9) / (time_us / 1e6)

    return ExperimentResult(time_us=time_us, gbps=gbps)


def print_results(experiments: List[Experiment]):
    headers = ["M", "N", "time_us", "gbps"]
    rows = [
        [
            e.config.m,
            e.config.n,
            round(e.result.time_us, 3),
            round(e.result.gbps, 3),
        ]
        for e in experiments
    ]
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
