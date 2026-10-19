"""LOCAL ADDITION for this repo (scripts/tabfm_experiments/row_weights.py).

Optional per-row weights for the context rows the preprocessing steps are fitted on.
While set, ``DifferentiableZNormStep`` fits weighted statistics (a row of weight w counts
like w copies of itself) and ``RemoveConstantFeaturesStep`` ignores rows of weight exactly
0, so that a weight of 0 is the same as deleting the row. Unset -- the normal case -- every
step fits exactly as before.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch

_CONTEXT_ROW_WEIGHTS: torch.Tensor | None = None


@contextmanager
def context_row_weights_installed(w: torch.Tensor | None) -> Iterator[None]:
    """Make ``w`` (one weight per context row, in the context's row order) visible to the
    steps for the duration of a fit; removed on the way out, also on an exception."""
    global _CONTEXT_ROW_WEIGHTS  # noqa: PLW0603
    previous = _CONTEXT_ROW_WEIGHTS
    _CONTEXT_ROW_WEIGHTS = w
    try:
        yield
    finally:
        _CONTEXT_ROW_WEIGHTS = previous


def context_row_weights(n_rows: int) -> torch.Tensor | None:
    """The installed weights, or None. A length other than ``n_rows`` means the step is
    being fitted on something other than the context rows (row subsampling, for one),
    which the weights cannot follow: that raises instead of silently misaligning."""
    w = _CONTEXT_ROW_WEIGHTS
    if w is not None and w.shape[0] != n_rows:
        raise ValueError(
            f"context row weights have {w.shape[0]} entries but the step is fitted on "
            f"{n_rows} rows"
        )
    return w
