"""E019 candidate-space definition.

How:
- Treat one experiment as a population search over qualitatively different policy families.
- Keep the first-stage population fixed at 48 arms so a single arena run can prune globally.

Why not:
- Do not encode one hypothesis per experiment. That recreates the slow local-search loop E019 replaces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateSpec:
    """How: one immutable arm records algorithm family and only search-relevant knobs."""

    candidate_id: str
    family: str
    base: str
    sell_margin: float | None = None
    route: tuple[str, ...] = ()
    phase_days: tuple[int, ...] = (0, 6, 12, 18, 24)
    opponent_conditioned: bool = False
    replay_prior: bool = False
    reward_mode: str = "terminal"
    entropy_coef: float = 0.01
    train_episodes: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["route"] = list(self.route)
        out["phase_days"] = list(self.phase_days)
        return out


FIXED_BASES = (
    "e008a",
    "e007",
    "e018d1",
    "e015ppo",
    "dense",
    "early",
    "dense_waterlock",
    "early_waterlock",
    "crop_wide",
    "animal_core",
    "animal_fertsell",
    "animal_fertuse",
)

PHASE_ROUTES = (
    ("early", "dense_waterlock", "crop_wide", "animal_core", "animal_fertuse"),
    ("early_waterlock", "crop_wide", "animal_core", "animal_fertsell", "animal_fertuse"),
    ("dense", "crop_wide", "animal_core", "animal_fertuse", "animal_fertuse"),
    ("crop_wide", "crop_wide", "animal_core", "animal_fertsell", "animal_fertuse"),
    ("animal_core", "animal_core", "animal_fertsell", "animal_fertuse", "animal_fertuse"),
    ("early", "animal_core", "animal_core", "animal_fertsell", "animal_fertuse"),
    ("dense_waterlock", "animal_core", "animal_fertuse", "animal_fertuse", "animal_fertuse"),
    ("crop_wide", "crop_wide", "crop_wide", "animal_core", "animal_fertsell"),
    ("early_waterlock", "dense_waterlock", "crop_wide", "crop_wide", "animal_fertuse"),
    ("dense", "dense_waterlock", "animal_core", "animal_core", "animal_fertsell"),
    ("crop_wide", "animal_core", "animal_fertuse", "animal_fertuse", "animal_fertsell"),
    ("early", "crop_wide", "crop_wide", "animal_fertsell", "animal_fertsell"),
)


def generate_candidate_specs() -> list[CandidateSpec]:
    """How: generate 48 diverse arms, not a dense local grid around one policy."""
    specs: list[CandidateSpec] = []

    # 12 anchors preserve all major policy families already built.
    for i, base in enumerate(FIXED_BASES):
        specs.append(CandidateSpec(
            candidate_id=f"F{i:02d}_{base}",
            family="fixed",
            base=base,
            notes="Existing policy anchor; no new learned routing.",
        ))

    # 12 trajectory-level mixtures test qualitatively different phase schedules.
    for i, route in enumerate(PHASE_ROUTES):
        specs.append(CandidateSpec(
            candidate_id=f"P{i:02d}",
            family="phase_router",
            base=route[0],
            route=route,
            notes="Five-phase route; switches only at day 0/6/12/18/24.",
        ))

    # 8 opponent-conditioned heuristics sweep broad reaction regimes.
    heuristic_bases = ("crop_wide", "animal_core", "animal_fertsell", "animal_fertuse")
    for i in range(8):
        specs.append(CandidateSpec(
            candidate_id=f"H{i:02d}",
            family="heuristic_router",
            base=heuristic_bases[i % len(heuristic_bases)],
            opponent_conditioned=True,
            sell_margin=(0.05 if i >= 4 else None),
            notes=f"Opponent-conditioned router regime={i}; {'with' if i >= 4 else 'without'} SELL overlay.",
        ))

    # 8 replay-family imitation routers: top replay family centroids drive expert selection.
    for i in range(8):
        specs.append(CandidateSpec(
            candidate_id=f"R{i:02d}",
            family="replay_router",
            base="animal_fertuse" if i % 2 else "crop_wide",
            opponent_conditioned=bool(i & 1),
            replay_prior=True,
            sell_margin=(None, 0.00, 0.05, 0.10)[i % 4],
            notes="Nearest top-replay strategy-family centroid chooses macro expert.",
        ))

    # 8 route-PPO arms: the action is an expert route, not a 720-turn primitive action.
    ppo_settings = (
        (False, False, "terminal", 0.01, 64),
        (True, False, "terminal", 0.01, 64),
        (False, True, "terminal", 0.01, 64),
        (True, True, "terminal", 0.01, 64),
        (True, True, "shaped", 0.01, 96),
        (True, True, "shaped", 0.03, 96),
        (True, True, "terminal", 0.03, 96),
        (True, True, "shaped", 0.00, 96),
    )
    for i, (opp_cond, replay_prior, reward_mode, entropy, budget) in enumerate(ppo_settings):
        specs.append(CandidateSpec(
            candidate_id=f"Q{i:02d}",
            family="route_ppo",
            base="animal_fertuse",
            opponent_conditioned=opp_cond,
            replay_prior=replay_prior,
            reward_mode=reward_mode,
            entropy_coef=entropy,
            train_episodes=budget,
            sell_margin=0.05 if i >= 4 else None,
            notes="PPO selects among macro experts at phase boundaries.",
        ))

    assert len(specs) == 48, len(specs)
    assert len({s.candidate_id for s in specs}) == 48
    return specs
