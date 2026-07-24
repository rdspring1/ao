# NVFP4 grouped GEMM for MoE training (SM100)

## Summary

This PR adds a **differentiable NVFP4 grouped GEMM** stack for Mixture-of-Experts
training on Blackwell (SM100+). It provides a single autograd entry point,
`_to_nvfp4_then_scaled_grouped_mm`, whose forward, dgrad, and wgrad each fuse a
per-group **randomized Hadamard transform (RHT)** with **NVFP4** quantization
(E2M1 4-bit codes + per-16 E4M3 block scales + a per-tensor scale) and feed
`torch.nn.functional.scaled_grouped_mm`.

Three grouped Triton kernels do the quantization work, all group-aware via
cumulative token-end offsets (jagged expert groups, 128-row aligned):

- `triton_group_rht_amax` — per-group row/col amax after RHT.
- `triton_group_rht_quantize_row_col` — fused rowwise + columnwise NVFP4 activation
  quantization.
- `triton_group_weight_quantize_2d` — per-expert 2D NVFP4 weight quantization
  (W and Wᵀ).

Additional features: `torch.compile(fullgraph=True)` support (fake registration
for `aten::_scaled_grouped_mm_v2`), **stochastic rounding** on gradients with
CUDA-graph-safe Philox state, optional token-group padding for unaligned groups,
DeepSeek-V3 benchmarks, and a single-GPU `GroupedExperts` example.

Verified on GB200 (SM100): 46 kernel tests + 19 end-to-end grouped-MM tests pass
(forward/backward SQNR, `torch.compile`, unaligned-token padding).

## Data flow

`_NVFP4GroupedMM` (`nvfp4_training/nvfp4_grouped_mm.py`) wires the kernels into an
`autograd.Function`:

**Forward** — quantize activations `A (M,K)`: `triton_group_rht_amax` → per-group
`(row_amax, col_amax)`, then `triton_group_rht_quantize_row_col` →
`(row_codes, row_sf, col_codes, col_sf)`. Quantize weights `W (E,N,K)`:
`triton_group_weight_quantize_2d` → `(W_codes, W_sf, Wᵀ_codes, Wᵀ_sf)`. Then the
grouped GEMM `A_row ⊗ Wᵀ → (M,N)`. The columnwise activation codes and `Wᵀ` codes
are saved for backward.

**Backward** — re-quantize `grad_output` (stochastic rounding **on**); **dgrad** =
`dY_row ⊗ Wᵀ`, **wgrad** = `dY_col ⊗ A_col` (using the saved forward codes).

Both GEMMs use scale recipe `[BlockWise1x16, TensorWise]` and swizzle
`[SWIZZLE_32_4_4, NO_SWIZZLE]`.

## Kernels

### 1. `triton_group_rht_amax` (`group_hadamard_amax_triton.py`)

Computes, per expert group `g`, `col_amax[g] = max|RHT(Aᵀ)|` and
`row_amax[g] = max|A|` without materializing the transformed output — these become
the per-tensor NVFP4 scales for the row/col quantizer.

- **Dual path.** A **persistent per-group-CTA** kernel (warp-specialized, TMA
  descriptor for the 16×16 Hadamard matrix, elementwise cumulative max with a
  single `atomic_max` per CTA) recovers single-tensor bandwidth for large groups;
  a **tiled** kernel handles small groups. The wrapper picks between them by
  average group size / SM occupancy.
- Group membership via binary search over cumulative token-end offsets;
  128-row-aligned boundaries so a tile never straddles two experts. Autotune key
  `["N"]` (per-tile occupancy tracks `N`, not total token count — keeps CUDA-graph
  capture warm under variable token counts).
- **Consumers:** forward (on `input_act`) and backward (on `grad_output`).

<details>
<summary>triton_group_rht_amax — DSV3 671B (E=128), effective memory bandwidth</summary>

| Projection | M | N | time (µs) | GB/s |
|-----------|---|---|-----------|------|
| gate/up (w1/w3) | 2048 | 7168 | 2068.51 | 1816.81 |
| down (w2) | 7168 | 2048 | 2064.38 | 1820.44 |
</details>

### 2. `triton_group_rht_quantize_row_col` (`group_rht_quantize_row_col_triton.py`)

Fuses two NVFP4 quantizations in one 128×128 tile per CTA:
- **Rowwise** — raw `A`, scaled by `row_amax[g]` → row FP4 codes + swizzled FP8
  scales (forward-GEMM LHS).
- **Columnwise** — RHT of `Aᵀ`, scaled by `col_amax[g]` → col FP4 codes + swizzled
  FP8 scales (saved for wgrad).

- Optional **Philox stochastic rounding** driven by a caller-owned 4-element
  `rng_state` (col/row seed+offset), advanced by the caller so it stays
  CUDA-graph-safe across replays.
- `num_warps` pinned at 4 (register-heavy body over-subscribes at 8); autotune key
  `["N", "STOCHASTIC_ROUNDING"]`.
- **Consumers:** forward (activations) and backward (`grad_output`, SR on).

<details>
<summary>triton_group_rht_quantize_row_col — DSV3 671B (E=128), effective memory bandwidth</summary>

Device peak memory bandwidth: 7928.1 GB/s (GB200)

| Projection | M | N | Rounding | time (µs) | GB/s | % peak |
|-----------|---|---|----------|-----------|------|--------|
| gate/up (w1/w3) | 2048 | 7168 | rtne | 2635.78 | 2227.82 | 28.10 |
| gate/up (w1/w3) | 2048 | 7168 | rs   | 5411.82 | 1085.04 | 13.69 |
| down (w2) | 7168 | 2048 | rtne | 2614.05 | 2246.33 | 28.33 |
| down (w2) | 7168 | 2048 | rs   | 5400.75 | 1087.26 | 13.71 |

`rtne` = round-to-nearest-even (forward path); `rs` = stochastic rounding
(backward path). SR roughly halves bandwidth due to the per-element Philox draws.
</details>

### 3. `triton_group_weight_quantize_2d` (`group_quantize_2d_triton.py`)

Per-expert 16×16-block NVFP4 quantization of both `W` and `Wᵀ` (no RHT), scaled by
a per-expert amax. Grid `(M//128, N//128, E)`, one tile per CTA, with **int64**
per-expert base pointers so large `E·M·N` weights address correctly. Autotune key
`["M", "N"]` (weight dims are fixed at model-definition time, so keying on them is
safe and re-tunes nothing per step). Produces the forward-GEMM RHS (`W`) codes and
the saved dgrad RHS (`Wᵀ`) codes.

<details>
<summary>triton_group_weight_quantize_2d — DSV3 671B (E=128), effective memory bandwidth</summary>

| Projection | M | N | time (µs) | GB/s |
|-----------|---|---|-----------|------|
| gate/up (w1/w3) | 2048 | 7168 | 2628.61 | 2233.89 |
| down (w2) | 7168 | 2048 | 2606.08 | 2253.20 |
</details>

> Benchmarks: `python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax`,
> `... bench_group_rht_quantize_row_col --rounding all`, `... bench_group_quantize_2d`.
> Shapes from `deepseek_v3_shapes.py`; 671B uses 256 experts at EP=2 → 128 local
> experts, `dim=7168`, `moe_hidden=2048`. GB/s is measured byte traffic (bf16 read +
> FP4 code writes + FP8 scale writes) over median kernel time.

## `GroupedExperts` — the TorchTitan integration vehicle

The kernels are consumed by a `GroupedExperts` `nn.Module` (reference in
`nvfp4_training/nvfp4_single_gpu_example.py`), which mirrors TorchTitan's
`torchtitan/models/moe/moe.py`. It holds expert weights `w1/w2/w3` and, in
`forward`, calls `_to_nvfp4_then_scaled_grouped_mm` for the gate, up, and down
projections with `SiLU(gate) * up`:

```python
gate = _to_nvfp4_then_scaled_grouped_mm(x, self.w1, sign_vector, sr_seed, offs=offsets)
up   = _to_nvfp4_then_scaled_grouped_mm(x, self.w3, sign_vector, sr_seed, offs=offsets)
hidden = F.silu(gate) * up
out  = _to_nvfp4_then_scaled_grouped_mm(hidden, self.w2, sign_vector, sr_seed, offs=offsets)
```

Input is pre-routed and packed by expert; `offsets = cumsum(num_tokens_per_expert)`
gives the jagged group boundaries.

Unlike the existing **MXFP8** path — which swaps in quantized grouped-mm via a
tensor subclass + `MoETrainingConfig`/`quantize_` op interception
(`--model.converters="quantize.grouped_mm.mx" --quantize.grouped_mm.mx.fqns="experts"`)
— NVFP4 currently integrates at the **functional-op level** inside
`GroupedExperts.forward`, because it needs an explicit RHT `sign_vector` and
stochastic-rounding `sr_seed` that don't fit the auto-quantize op-swap. A future
`NVFP4…OpConfig` wrapper could follow the MXFP8 subclass pattern if transparent
`quantize_(model, ...)` integration is desired.

## Testing

On GB200 (SM100):
- `test/prototype/moe_training/nvfp4_training/` — 46 kernel tests (RHT amax,
  row/col quantize, 2D weight quantize) incl. oracle SQNR/ULP parity, padded-capacity
  masking, stochastic-rounding determinism.
- `test/prototype/moe_training/test_nvfp4_grouped_mm.py` — 19 tests: forward/backward
  SQNR vs bf16, `torch.compile(fullgraph=True)` fwd/bwd, unaligned-token padding.
