"""How: DAgger/counterfactual rolloutを再現可能な学習行へ固定する。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CounterfactualBatch:
    """How: learner訪問状態と実測option returnを同じ単位で保持する。"""

    history: np.ndarray
    valid_lengths: np.ndarray
    option_features: np.ndarray
    option_returns: np.ndarray
    observed_mask: np.ndarray
    compatible_mask: np.ndarray
    behavior_option: np.ndarray

    def validate(self) -> None:
        n = self.history.shape[0]
        if self.valid_lengths.shape != (n,):
            raise ValueError("valid_lengths shape")
        if self.option_features.shape[:2] != self.option_returns.shape:
            raise ValueError("option shape")
        if self.option_returns.shape != self.observed_mask.shape or self.option_returns.shape != self.compatible_mask.shape:
            raise ValueError("option masks")
        if self.behavior_option.shape != (n,):
            raise ValueError("behavior_option shape")
        if not np.all(self.compatible_mask.any(axis=1)):
            raise ValueError("no compatible option")
