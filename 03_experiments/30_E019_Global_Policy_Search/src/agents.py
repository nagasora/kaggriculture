"""Executable policy families for the E019 global-search arena.

How:
- Reuse proven E015/E018 executors and change only routing/overlay logic.
- Construct a fresh agent for every game so state cannot leak across seeds.

Why not:
- Do not rewrite the farm simulator or low-level worker planner inside E019.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from .candidate_space import CandidateSpec


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AssetContext:
    def __init__(self, e015_root: Path, e018_agent_dir: Path, replay_model_dir: Path | None = None):
        self.e015_root = Path(e015_root)
        self.e018_agent_dir = Path(e018_agent_dir)
        self.replay_model_dir = Path(replay_model_dir) if replay_model_dir else None
        root = str(self.e015_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from e014b3.runner import make_agent  # type: ignore
        from e015.macro import PolicyAgent  # type: ignore
        self.make_e015_agent = make_agent
        self.PolicyAgent = PolicyAgent
        self.e015_policy_payload = json.loads((self.e015_root / "runs/pilot/policy.json").read_text())

    def fixed(self, name: str, seed: int = 0):
        if name == "e015ppo":
            return self.PolicyAgent(self.e015_policy_payload, deterministic=True, seed=seed)
        if name in {"e007", "e018d1"}:
            local = str(self.e018_agent_dir)
            if local not in sys.path:
                sys.path.insert(0, local)
            filename = "e007_base.py" if name == "e007" else "main.py"
            return _load_file_module(f"e019_{name}_{seed}_{id(self)}", self.e018_agent_dir / filename).agent
        return self.make_e015_agent(name)


class PhaseRouterAgent:
    def __init__(self, ctx: AssetContext, route: tuple[str, ...], phase_days: tuple[int, ...], seed: int = 0):
        if len(route) != len(phase_days): raise ValueError("route/phase_days mismatch")
        self.ctx, self.route, self.phase_days, self.seed = ctx, route, phase_days, int(seed)
        self.phase_index, self.inner = -1, None

    def __call__(self, obs: dict, configuration: Any = None) -> dict:
        day = int(obs["step"]) // 24
        idx = max(i for i, start in enumerate(self.phase_days) if day >= start)
        if idx != self.phase_index:
            self.phase_index = idx
            self.inner = self.ctx.fixed(self.route[idx], self.seed + idx * 1009)
        return self.inner(obs, configuration)


class HeuristicRouterAgent:
    def __init__(self, ctx: AssetContext, regime: int, fallback: str, seed: int = 0):
        self.ctx, self.regime, self.fallback, self.seed = ctx, int(regime), fallback, int(seed)
        self.phase, self.inner = -1, None

    @staticmethod
    def _public(farm: dict) -> tuple[int, int, int]:
        tiles = [t for row in farm["tiles"] for t in row]
        productive = sum(isinstance(t, dict) and ("crop" in t or "animal" in t) for t in tiles)
        animals = sum(isinstance(t, dict) and "animal" in t for t in tiles)
        return len(farm["unlocked_quadrants"]), productive, animals

    def _choose(self, obs: dict) -> str:
        p = int(obs["player"]); own = self._public(obs["farms"][p]); opp = self._public(obs["farms"][1-p]); day = int(obs["step"])//24
        aggressive = opp[0] > own[0] or opp[1] > own[1] + (8 if self.regime % 2 else 15)
        animal_meta = opp[2] >= (4 + self.regime % 4)
        if day <= 6 and aggressive: return "early_waterlock" if self.regime % 2 else "early"
        if animal_meta: return "animal_fertuse" if self.regime % 3 else "animal_fertsell"
        if day >= 18: return "animal_fertuse" if own[2] >= 4 else "crop_wide"
        return self.fallback

    def __call__(self, obs: dict, configuration: Any = None) -> dict:
        phase = min(4, int(obs["step"]) // 144)
        if phase != self.phase:
            self.phase = phase; self.inner = self.ctx.fixed(self._choose(obs), self.seed + phase * 1291)
        return self.inner(obs, configuration)


class ReplayRouterAgent:
    def __init__(self, ctx: AssetContext, fallback: str, opponent_conditioned: bool, seed: int = 0):
        if ctx.replay_model_dir is None: raise ValueError("replay_model_dir is required")
        self.ctx, self.fallback, self.opponent_conditioned, self.seed = ctx, fallback, bool(opponent_conditioned), int(seed)
        self.phase, self.inner = -1, None
        self.profiles = pd.read_csv(ctx.replay_model_dir / "day_family_profiles.csv")

    @staticmethod
    def _row(obs: dict) -> dict[str, float]:
        p = int(obs["player"]); farm = obs["farms"][p]; opp = obs["farms"][1-p]
        tiles = [t for row in farm["tiles"] for t in row]; opp_tiles = [t for row in opp["tiles"] for t in row]
        out = {
            "money": float(farm["money"]), "opp_money": float(opp["money"]), "money_margin": float(farm["money"]-opp["money"]),
            "unlocked_quadrants": float(len(farm["unlocked_quadrants"])),
            "productive_tiles": float(sum(isinstance(t, dict) and ("crop" in t or "animal" in t) for t in tiles)),
            "total_stock": float(sum((obs.get("private") or {}).get("shed", {}).values())),
            "shop_count": float(len((obs.get("town") or {}).get("unlocked_shops", []))),
            "opp_unlocked_quadrants": float(len(opp["unlocked_quadrants"])),
            "opp_productive_tiles": float(sum(isinstance(t, dict) and ("crop" in t or "animal" in t) for t in opp_tiles)),
        }
        for crop in ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON"):
            out[f"crop_{crop}"] = float(sum(isinstance(t,dict) and t.get("crop")==crop for t in tiles))
        for animal in ("COW","SHEEP","GOOSE"):
            out[f"animal_{animal}"] = float(sum(isinstance(t,dict) and t.get("animal")==animal for t in tiles))
        return out

    def _choose(self, obs: dict) -> str:
        day = int(obs["step"])//24; days = np.asarray(sorted(self.profiles.day.unique()), dtype=int)
        part = self.profiles[self.profiles.day.eq(int(days[np.argmin(np.abs(days-day))]))].copy(); row = self._row(obs)
        cols = [c for c in row if c in part.columns]
        if not self.opponent_conditioned: cols = [c for c in cols if not c.startswith("opp_") and c != "money_margin"]
        if not cols: return self.fallback
        arr = part[cols].to_numpy(dtype=float); x = np.asarray([row[c] for c in cols], dtype=float); scale = np.nanstd(arr, axis=0); scale[scale<1e-6]=1.0
        chosen = part.iloc[int(np.argmin(np.square((arr-x)/scale).mean(axis=1)))]
        return str(chosen.get("expert_prior") or self.fallback)

    def __call__(self, obs: dict, configuration: Any = None) -> dict:
        phase = min(4, int(obs["step"])//144)
        if phase != self.phase:
            self.phase = phase; self.inner = self.ctx.fixed(self._choose(obs), self.seed + phase*1877)
        return self.inner(obs, configuration)


class GenericSellOverlay:
    def __init__(self, ctx: AssetContext, base: Callable, margin: float, seed: int = 0):
        self.base = base; local = str(ctx.e018_agent_dir)
        if local not in sys.path: sys.path.insert(0, local)
        residual = _load_file_module(f"e019_residual_{seed}_{id(self)}", ctx.e018_agent_dir / "e018_sell_residual_v2.py")
        self.policy = residual.SellResidualPolicy.from_joblib(
            str(ctx.e018_agent_dir / "e018c_sell_binary_full_supported.joblib"),
            str(ctx.e018_agent_dir / "e018c_sell_fraction_full_supported.joblib"), confidence_margin=float(margin))

    @staticmethod
    def _merge(action: dict, proposals: list[Any], prices: dict) -> dict:
        if not proposals: return action
        market = copy.deepcopy(action.get("market") or [])[:10]
        existing = {str(o[1]): i for i,o in enumerate(market) if o and len(o)>=3 and o[0]=="SELL"}
        ranked = sorted(proposals, key=lambda p: p.confidence_excess*max(1,int(prices.get(p.product,1)))*p.quantity, reverse=True)
        new_orders = 0
        for p in ranked:
            if p.product in existing: market[existing[p.product]][2] = int(market[existing[p.product]][2]) + int(p.quantity)
            elif len(market)<10 and new_orders<2:
                market.append(["SELL",p.product,int(p.quantity)]); existing[p.product]=len(market)-1; new_orders+=1
        out = copy.deepcopy(action); out["market"] = market; return out

    def __call__(self, obs: dict, configuration: Any = None) -> dict:
        action = self.base(obs, configuration); step = int(obs["step"])
        if step < 576 or step >= 719: return action
        try:
            shed = dict((obs.get("private") or {}).get("shed", {}))
            proposals = self.policy.propose(obs, list(action.get("market") or []), available_stock=shed)
            return self._merge(action, proposals, (obs.get("market") or {}).get("prices", {}))
        except Exception:
            # Why not: an overlay error must never invalidate a viable underlying expert action.
            return action


def build_candidate(spec: CandidateSpec, ctx: AssetContext, seed: int, route_policy_payload: dict | None = None):
    if spec.family == "fixed": base = ctx.fixed(spec.base, seed)
    elif spec.family == "phase_router": base = PhaseRouterAgent(ctx, spec.route, spec.phase_days, seed)
    elif spec.family == "heuristic_router": base = HeuristicRouterAgent(ctx, int(spec.candidate_id[1:]), spec.base, seed)
    elif spec.family == "replay_router": base = ReplayRouterAgent(ctx, spec.base, spec.opponent_conditioned, seed)
    elif spec.family == "route_ppo":
        if route_policy_payload is None:
            base = ReplayRouterAgent(ctx, spec.base, spec.opponent_conditioned, seed) if spec.replay_prior else ctx.fixed(spec.base, seed)
        else:
            from .route_ppo import RoutePolicyAgent
            base = RoutePolicyAgent(ctx, route_policy_payload, seed=seed, opponent_conditioned=spec.opponent_conditioned)
    else: raise ValueError(spec.family)
    return GenericSellOverlay(ctx, base, spec.sell_margin, seed) if spec.sell_margin is not None and spec.base != "e018d1" else base
