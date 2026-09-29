"""How: 学習selectorをE043のcoherent plan上に安全に載せるための昇格ゲート。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .macro import BLOCK_TURNS, macro_boundary


@dataclass(frozen=True)
class SelectorConfig:
    """How: early routingとlate residualを分離し、prefix破壊を構造的に禁止する。"""

    block_turns: int = BLOCK_TURNS
    route_switch_last_step: int = 144
    min_q_margin: float = 0.03
    base_option: int = 0


class SafeOptionSelector:
    """How: Qが十分優位で、かつ工程互換なmacro境界でだけbase routeから離れる。"""

    def __init__(self, config: SelectorConfig = SelectorConfig()) -> None:
        self.config = config
        self.current_option = config.base_option

    def choose(self, step: int, q: Sequence[float], compatible: Sequence[bool]) -> int:
        if len(q) != len(compatible):
            raise ValueError("q/compatible length mismatch")
        if not macro_boundary(step, self.config.block_turns):
            return self.current_option
        if step > self.config.route_switch_last_step:
            # Why not: day6以降のfull-route switchはE042で工程因果を壊した。
            self.current_option = self.config.base_option
            return self.current_option
        valid = np.asarray(compatible, dtype=bool)
        if not valid.any():
            self.current_option = self.config.base_option
            return self.current_option
        values = np.asarray(q, dtype=float).copy()
        values[~valid] = -np.inf
        order = np.argsort(-values)
        best = int(order[0])
        second = float(values[order[1]]) if len(order) > 1 and np.isfinite(values[order[1]]) else -np.inf
        base = self.config.base_option
        baseline = float(values[base]) if 0 <= base < len(values) and valid[base] else second
        if not np.isfinite(values[best]) or values[best] - baseline < self.config.min_q_margin:
            best = base if 0 <= base < len(values) and valid[base] else best
        self.current_option = best
        return best
