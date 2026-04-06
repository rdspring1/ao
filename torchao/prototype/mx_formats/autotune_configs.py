"""Autotuning infrastructure for RHT Triton kernels.

The cache lives for the process lifetime. First call for a given (kernel, shape, device)
triggers benchmarking across all configs; subsequent calls are a dict lookup.

CUDA graphs: call each wrapper once before graph capture to warm up the autotuner.
Subsequent calls (including those inside a captured graph) are CUDA graph safe.
"""
import dataclasses
import os
from typing import Callable

import torch


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    BLOCK_M: int
    BLOCK_N: int
    NUM_STAGES: int
    NUM_WARPS: int


# Process-lifetime cache: maps (kernel_name, *shape_dims, device_index) -> KernelConfig
_autotune_cache: dict[tuple, KernelConfig] = {}


def do_bench(fn: Callable, warmup_iters: int = 3, bench_iters: int = 10) -> float:
    """Benchmark a zero-argument callable; return median elapsed time in milliseconds."""
    for _ in range(warmup_iters):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times: list[float] = []
    for _ in range(bench_iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def get_best_config(
    cache_key: tuple,
    configs: list[KernelConfig],
    benchmark_fn: Callable[[KernelConfig], None],
) -> KernelConfig:
    """Return the fastest config for cache_key, benchmarking if not cached.

    Args:
        cache_key: Hashable key encoding (kernel_name, *shape, device_index).
        configs: Candidate configs to evaluate.
        benchmark_fn: Called as benchmark_fn(cfg) for each config; must launch the
            kernel on pre-allocated scratch tensors so it is side-effect-free.

    Returns:
        The KernelConfig with the lowest median launch time.

    Note:
        First call compiles and benchmarks all configs (expect JIT latency per config).
        Subsequent calls with the same cache_key return immediately from cache.
    """
    if cache_key in _autotune_cache:
        return _autotune_cache[cache_key]

    best_cfg: KernelConfig | None = None
    best_time = float("inf")

    for cfg in configs:
        try:
            # Redirect stderr at fd level to suppress MLIR compiler noise from invalid configs.
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
            try:
                elapsed = do_bench(lambda cfg=cfg: benchmark_fn(cfg))
            finally:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
        except Exception:
            # Config may be invalid for this shape (e.g. tile larger than tensor).
            continue

        if elapsed < best_time:
            best_time = elapsed
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError(
            f"All configs failed for cache_key={cache_key}. "
            "Check that the tensor shape is compatible with at least one config."
        )

    _autotune_cache[cache_key] = best_cfg
    return best_cfg
