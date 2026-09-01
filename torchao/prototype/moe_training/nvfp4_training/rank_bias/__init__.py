"""Standalone NVFP4 QDQ rank-bias analysis, ported from kitchen.

Ported from kitchen's ``experimental/tensor_dump_analysis/analyze_rank_bias.py``
and bitwise-verified against a built kitchen and the psx ``clippy`` kernels. It
calls no torchao kernels; it lives here only so that the script ships inside the
container image and one revision pins it (see ``analyze_rank_bias`` for why).

Deliberately empty of imports: ``analyze_rank_bias`` pulls matplotlib and
``nvfp4_cutedsl`` pulls the CuTe DSL, and neither should be a cost of importing
``torchao.prototype.moe_training.nvfp4_training``.
"""
