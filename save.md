# Save Note

## Goal
Make `nvfp4_mm_triton` fully correct under `torch.compile(mode="reduce-overhead", fullgraph=True)` CUDA graphs, including verified SR diversity in the backward pass.

## Current State
Forward CUDA graph path works. Four issues remain, two blocking:

### Blocking

**1. `offset_base` not in `mutates_args` — SR offset frozen under CUDA graphs (fix incomplete)**
`hadamard_quantize_row_col_triton.py:293`: `offset_base` is `Optional[torch.Tensor]` with a `torch.randint` fallback inside the custom_op body. `seed_buf` was fixed with `mutates_args=("seed_buf",)` and made a required arg; `offset_base` was not. In CUDA graph mode, the offset is pool-allocated at capture time and frozen across replays. Philox SR requires both seed AND offset to vary — the fix is half-applied.

Fix: make `offset_base` a required `offset_buf: torch.Tensor`, add to `mutates_args=("seed_buf", "offset_buf")`, pre-allocate at all call sites.

**2. Backward SR seeds not pre-allocated (correctness risk — unverified)**
`nvfp4_linear.py:298–305` generates `dy_sr_seed` and `dy_sr_offset` via `torch.randint` inside the compiled `backward` body. Both pool-allocate at graph-capture time. Backward SR diversity is probably frozen under `reduce-overhead`. Fix requires `dy_sr_seed_buf`/`dy_sr_offset_buf` stored as persistent state outside `torch.compile`, mutated before each backward call.

**3. `triton_rht_amax` and `triton_weight_quantize_2d` no longer self-install TMA allocator**
Both had `set_allocator(lambda ...: torch.empty(...))` removed. The new approach relies on `get_tma_workspace` having been called first to set the global allocator. Direct eager callers that skip `get_tma_workspace` will fail with a TMA allocator error. Existing tests calling `triton_weight_quantize_2d` directly (`test_quantize_2d_triton.py` non-graph tests) may now depend on test execution order. Previously each function was self-contained.

### Non-blocking

**4. Stale docstring**
`hadamard_quantize_row_col_triton.py:231` references `philox_seed_base` but the parameter no longer exists — replaced by `seed_buf`.

**5. `_fwd_seed_buf` allocated inside compiled forward**
`nvfp4_linear.py:234`: `torch.empty` inside compiled region is harmless (SR=False, kernel never reads it), but unnecessary and misleading.

**6. `get_tma_workspace` is misnamed**
Does two things: allocates TMA scratch buffer AND warms `get_rht_matrix` lru_cache. Should be renamed `prepare_for_cuda_graph(device)`.

## TODOs
- [ ] **[BLOCKING]** Fix `offset_buf`: required arg, `mutates_args=("seed_buf", "offset_buf")`, pre-allocate at all call sites
- [ ] **[BLOCKING]** Fix backward SR: confirm freeze with `test_nvfp4_linear.py`, then store `dy_sr_seed_buf`/`dy_sr_offset_buf` as persistent module state
- [ ] **[BLOCKING]** Restore TMA allocator self-install for direct eager callers of `triton_rht_amax` and `triton_weight_quantize_2d` (or enforce `prepare_for_cuda_graph` contract)
- [ ] Fix stale `philox_seed_base` docstring in `triton_rht_quantize_row_col`
- [ ] Remove `_fwd_seed_buf` allocation from compiled forward (or make it module state)
- [ ] Rename `get_tma_workspace` → `prepare_for_cuda_graph`

## Next Action
Fix `offset_buf` first: make it a required `offset_buf: torch.Tensor`, add to `mutates_args=("seed_buf", "offset_buf")`, update `register_fake`, update all call sites in `nvfp4_linear.py` and tests.

## Why
Mechanical, self-contained, highest-confidence fix. SR correctness requires both seed AND offset to vary — fixing only one is incomplete. The backward persistent-state fix is harder and needs confirmation first; the TMA allocator issue needs a design decision.

## Expected Outcome
SR outputs vary per replay on both seed and offset dimensions. A CUDA graph SR diversity test asserting `r1 != r2` should pass for both axes.

## Confidence
HIGH (offset_buf fix correctness) / MEDIUM (backward SR diversity unverified)

## Risk
LOW for offset_buf fix — mechanical, mirrors the seed_buf fix exactly. MEDIUM for backward fix — requires persistent state.

## Evidence
- Forward CUDA graph tests all pass: `test_triton_rht_quantize_row_col_cuda_graph_compile`, `test_triton_weight_quantize_2d_colwise_cuda_graph_compile`, `test_nvfp4_mm_triton_cuda_graph_compile`
- `test_hadamard_quantize_row_col_triton.py:206–225`: pre-allocated `seed_buf` + mutation is the correct pattern — `offset_buf` needs the same treatment
- `hadamard_quantize_row_col_triton.py:293–295`: `offset_base` fallback `torch.randint` inside custom_op — pool-allocates, not in `mutates_args`
- `nvfp4_linear.py:298–305`: `dy_sr_seed`/`dy_sr_offset` via `randint` inside compiled body
- `hadamard_amax_triton.py` and `quantize_2d_triton.py`: `set_allocator` removed — global TMA allocator now an implicit precondition

## High-value tests needed
- `triton_rht_amax` called directly in eager mode without prior `get_tma_workspace` — verifies no TMA allocator regression for direct callers
- Backward SR diversity: assert compiled backward produces different `dy_col` across two calls with same input — verifies backward SR is not frozen

### Best Next Mode
Implementer — `offset_buf` fix is the highest-priority mechanical change.
