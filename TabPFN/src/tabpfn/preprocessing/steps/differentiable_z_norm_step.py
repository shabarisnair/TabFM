#  Copyright (c) Prior Labs GmbH 2026.

"""Differentiable Z-Norm Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import torch

from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)
from tabpfn.preprocessing.steps.context_row_weights import context_row_weights

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureSchema


class DifferentiableZNormStep(PreprocessingStep):
    """Differentiable Z-Norm Step."""

    def __init__(self):
        super().__init__()

        self.means = torch.tensor([])
        self.stds = torch.tensor([])

    @override
    def _fit(  # type: ignore
        self,
        X: torch.Tensor,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        # LOCAL ADDITION (scripts/tabfm_experiments/row_weights.py): with context row
        # weights w installed, mean = sum w x / sum w and var = sum w (x - mean)^2 /
        # (sum w - 1) -- a row of weight w counts like w copies, so w = 1 everywhere is
        # the unbiased X.std below and w = 0 is the row deleted.
        w = context_row_weights(X.shape[0])
        if w is None:
            self.means = X.mean(dim=0, keepdim=True)
            self.stds = X.std(dim=0, keepdim=True)
        else:
            w_N1 = w.to(device=X.device, dtype=X.dtype).reshape(-1, 1)
            total = w_N1.sum()
            self.means = (w_N1 * X).sum(dim=0, keepdim=True) / total
            var = (w_N1 * (X - self.means) ** 2).sum(dim=0, keepdim=True) / (total - 1)
            self.stds = var.sqrt()
        return feature_schema

    @override
    def _transform(  # type: ignore
        self, X: torch.Tensor, *, is_test: bool = False
    ) -> tuple[torch.Tensor, None, None]:
        assert X.shape[1] == self.means.shape[1]
        assert X.shape[1] == self.stds.shape[1]
        return (X - self.means) / self.stds, None, None


__all__ = [
    "DifferentiableZNormStep",
]
