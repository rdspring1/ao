# Debug Session: TE E=4 Sequential Timing

## Experiment 1

- Hypothesis: Four validated TE single-expert calls can provide a same-work E=4 device-time baseline for grouped TorchAO kernels.
- Action: Ran the targeted TE-default timing script from `/home/me/nvfp4`.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` with the E=4 timing script shown in the session log.
- Result: Failed during imports with `ModuleNotFoundError: No module named 'benchmarks.prototype'`; no benchmark ran.
- Interpretation: The torchao `benchmarks` package is checkout-local and not editable-installed. Add `/home/me/nvfp4/third_party/torchao` to `sys.path`, matching the existing TE harness.
- Canonical-command status: The existing harness performs this path insertion explicitly.
- Failure classification: invocation mismatch.

## Experiment 2

- Hypothesis: Adding the torchao checkout to `sys.path` will make the same E=4 TE timing experiment runnable.
- Action: Re-ran the same script with `/home/me/nvfp4/third_party/torchao` prepended to `sys.path`.
- Result: Passed. Gate/up: RHT 59.210 us, 2D weight 60.489 us. Down: RHT 59.086 us, 2D weight 60.617 us.
- Interpretation: Four validated TE-default single-expert calls provide a same-work E=4 fallback baseline. It is not TE's grouped API and must remain labeled `TE 4xsingle`.
- Canonical-command status: Matches the path setup and timing helper used by the existing TE harness.
- Failure classification: resolved invocation mismatch.

## Experiment 3

- Top hypothesis: The grouped TE harness passes row-prefix `tensor_offsets`, but TE requires element-prefix offsets (`cumsum(first_dims * logical_last_dim)`). Source: TE `common.h` and `ep.py` use `tex.splits_to_offsets(first_dims, hidden)`.
- Strongest competing hypothesis: Grouped and single TE quantizers use different RHT sign masks.
- Untested assumption: Correct element offsets make E=1 grouped TE reproduce its single-tensor output byte-for-byte.
- Best next experiment: Compare E=1 grouped and single TE outputs with `tensor_offsets=tex.splits_to_offsets(first_dims, hidden)`.
- Missing source data: none.

### Result

With `tensor_offsets=tex.splits_to_offsets(first_dims, hidden)`, E=1 grouped TE matched
single-tensor TE exactly for row/column codes and scales. E=4 grouped TE then matched both
TorchAO backends with 0% differing bytes across the same four outputs. The competing sign
mask hypothesis is rejected.

## Experiment 4

- Hypothesis: Corrected grouped metadata enables a valid same-process E=4 timing comparison.
- Action: Timed grouped Triton, grouped CuteDSL, and grouped TE default on both 671B shapes.
- Result: Gate/up 92.820 / 62.367 / 47.930 us; down 92.297 / 64.122 / 48.003 us.
- Interpretation: Grouped TE is valid and is the RHT comparator. Four single TE calls remain necessary only for 2D weight quantization.
- Failure classification: resolved harness bug.
