# Copyright © 2026 Apple Inc.
# The fork predates upstream's layers/ layout: the MoE layers still live in
# models/switch_layers.py. This re-export lets the Inkling 2 port keep its
# upstream imports unchanged.

from mlx_lm.models.switch_layers import (  # noqa: F401
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
    _gather_sort,
    _scatter_unsort,
)
