"""Layer reset: restore a layer's weights and re-derive its crossbar
conductances the same way MemTorch derived them originally.

Fixes the original codebase's bug #5: `crossbar.conductance_matrix.copy_(
module.weight.data)` copied the raw weight tensor directly into the
conductance matrix, ignoring the actual weight->conductance mapping
(naive_map) and Scheme.DoubleColumn's split into two separate crossbars with
a different shape than the weight tensor - it silently no-op'd behind a bare
`except Exception: print(...)`.

Here, failure is never silent: `apply_layer_reset` returns False and logs
the real exception; it never reports success without having actually
rewritten every crossbar handle for the layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


def apply_layer_reset(
    layer: torch.nn.Module,
    handles: list[CrossbarHandle],
    original_weight: torch.Tensor,
    mapping_routine: Callable,
    r_on: float,
    r_off: float,
    scheme: str = "DoubleColumn",
) -> bool:
    """Restore `layer.weight` to `original_weight`, then re-derive each
    handle's conductance matrix via the same mapping_routine patch_model
    used, respecting DoubleColumn's positive/negative crossbar split.

    Returns False (and logs) on any failure - never swallows an exception
    and reports success anyway.
    """
    try:
        layer.weight.data.copy_(original_weight)

        mapped = mapping_routine(layer.weight.data, r_on, r_off, scheme)
        # naive_map returns a single tensor for SingleColumn, a
        # (positive, negative) tuple of tensors for DoubleColumn.
        matrices = mapped if isinstance(mapped, (tuple, list)) else (mapped,)

        if len(matrices) != len(handles):
            raise ValueError(
                f"mapping_routine produced {len(matrices)} matrices but layer has "
                f"{len(handles)} crossbar handles (scheme={scheme})"
            )

        for handle, matrix in zip(handles, matrices):
            handle.accessor.write(matrix)

        logger.info("Reset layer with %d crossbar(s) to original weights", len(handles))
        return True

    except Exception:
        logger.exception("Layer reset failed - weights/conductances left in their prior state")
        return False
