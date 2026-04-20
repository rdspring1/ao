# SUMMARY

Mode: Implementer
Status: STOPPED

## Goal
Make `sr_offset` a live graph input to the compiled backward so SR diversity works across steps under compiled autograd + CUDA graphs.

## Current state
Both "module buffer" and "explicit forward argument" approaches fail identically (g1 == g2).

Root cause (confirmed): compiled autograd's `ctx.save_for_backward(sr_offset)` creates a frozen snapshot of the tensor at CUDA graph capture time. The backward CUDA graph's `sr_offset` input is fixed at the capture-time value regardless of:
- whether `sr_offset` is a module buffer or an explicit forward argument
- whether `sr_offset.add_(1)` is called in-place before the next call

The explicit-argument fix helps for the FORWARD graph (torch.compile tracks it as a live input). But the BACKWARD graph is compiled separately by compiled autograd, and its inputs come from `ctx.saved_tensors` — which are snapshots, not live pointers to the original tensors.

## What I want to do next
Determine whether `sr_offset` can reach the backward WITHOUT going through `ctx.save_for_backward`. Options:

**Option A — Custom op backward**: convert `nvfp4_mm_triton` from `torch.autograd.Function` to a `torch.library.custom_op` with an explicit backward registered via `torch.library.register_autograd`. The backward would take `sr_offset` as a direct op input (not a saved tensor). Whether this makes `sr_offset` a live backward input under compiled autograd is unknown.

**Option B — Remove sr_offset from backward entirely**: have the backward recompute a seed from fixed tensors only (e.g., a per-step counter tensor that is a direct output of some upstream op), sidestepping the save_for_backward issue.

**Option C — Accept the limitation for compiled autograd**: the manual-CUDA-graph pattern (forward + backward without compiled autograd) may work correctly because there the forward CUDA graph and backward CUDA graph are separate captures, and explicit-arg updates propagate. The compiled-autograd path may be an unsupported scenario. Drop or re-scope the diversity test.

**Option D — Use torch hooks / post-backward hooks** to update sr_offset inside the compiled backward in a way that compiled autograd recognizes as a live input update (needs PyTorch internals knowledge).

## Why
The `save_for_backward` freeze is a fundamental constraint of how compiled autograd captures backward CUDA graphs. There is no workaround within `torch.autograd.Function`. The solution requires either restructuring the backward (A, B) or scoping the guarantee (C).

## Expected outcome
If Option C: eager + manual CUDA graph diversity test passes; compiled-autograd test is removed or explicitly scoped to "no SR diversity guarantee."
If Option A: needs expert knowledge of `torch.library.custom_op` backward registration + compiled autograd behavior.

## Confidence: LOW
The save_for_backward freeze has been confirmed. No remaining option is clearly correct without PyTorch internals expertise.

## Risk: LOW for Option C, MEDIUM for A/B
Option C is a documentation/test scope change only. A and B require significant code restructuring.

## Evidence
- g1 == g2 regardless of whether sr_offset is a module buffer or explicit arg
- `test_nvfp4_mm_triton_cuda_graph_compile` (forward-only) passes → forward graph is fine
- Both module-buffer and explicit-argument approaches give identical failures → save_for_backward is the freeze point
- The OLD design kept `sr_offset.add_(1)` INSIDE the backward graph → compiled autograd replayed the mutation → diversity worked as a side effect of in-graph mutation (not by reading a live external value)

## What changed
- `torchao/prototype/mx_formats/nvfp4_linear.py`: `Nvfp4Linear` — removed `sr_offset` buffer, `advance_sr_offset()`, added `sr_offset` as explicit `forward()` argument
- `test/prototype/mx_formats/test_nvfp4_tensor.py`: updated test to pass `sr_offset` explicitly to `compiled_layer(x, sr_offset)` — STILL FAILS

## What failed or blocked progress
- `ctx.save_for_backward(sr_offset)` in compiled autograd creates a frozen snapshot
- Both module-buffer and explicit-argument designs fail identically because both ultimately route `sr_offset` to the backward via `ctx.saved_tensors`

## Missing context
- Whether `torch.library.custom_op` backward registration exposes saved-tensor inputs as live graph inputs under compiled autograd
- Whether PyTorch has a mechanism to mark specific saved tensors as "live" (bypass the freeze)
- Whether the manual-CUDA-graph path (without compiled_autograd) actually works correctly with the explicit-argument design

## Needs from user
Decision on which option to pursue (A, B, C, or D above). Specifically:
- Is compiled autograd SR diversity a hard requirement, or is eager/manual-CUDA-graph the primary target?
- Is it acceptable to remove the compiled-autograd diversity test and scope the guarantee to eager + manual CUDA graph?

## Best next mode: Debugger
Confirm whether option C (manual CUDA graph) actually works with the explicit-argument design before committing to a direction.
