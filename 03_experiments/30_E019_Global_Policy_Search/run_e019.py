"""E019 one-command global policy search.

How:
1. Discover replay strategy families from the full E018 daily macro corpus.
2. Train all route-PPO arms under small, diverse budgets.
3. Run 48 -> 12 -> 4 successive-halving rounds on the fast reconstructed simulator.
4. Re-rank only the top four in the exact Kaggle environment and promote three submissions.

Why not:
- Do not stop between sub-hypotheses for manual approval. The arena is the experiment unit.
- Do not trust the reconstructed simulator as the final selector; it is deliberately only a broad-search screen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.arena import run_round
from src.candidate_space import generate_candidate_specs
from src.official_gate import run_official_gate
from src.replay_families import fit_families
from src.route_ppo import replay_prior_from_profiles, train_route_ppo
from src.submission_builder import build_submission


def run(args: argparse.Namespace) -> dict:
    out = args.out
    results = out / "results"
    models = out / "models"
    reports = out / "reports"
    for p in (out, results, models, reports):
        p.mkdir(parents=True, exist_ok=True)

    specs = generate_candidate_specs()
    (out / "candidate_specs.json").write_text(
        json.dumps([s.to_dict() for s in specs], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    family_dir = models / "replay_families"
    summary_path = family_dir / "family_summary.json"
    if not summary_path.exists() or args.refit_replay_families:
        family_summary = fit_families(args.replay_csv, family_dir)
    else:
        family_summary = json.loads(summary_path.read_text())

    # How: learning/no-learning is part of the same initial population, not a later experiment branch.
    for idx, spec in enumerate(s for s in specs if s.family == "route_ppo"):
        model_path = models / f"{spec.candidate_id}.json"
        if model_path.exists() and not args.retrain_ppo:
            continue
        prior = replay_prior_from_profiles(family_dir / "family_profiles.csv") if spec.replay_prior else None
        print(f"[E019] training {spec.candidate_id}: episodes={spec.train_episodes} replay_prior={spec.replay_prior} opp_cond={spec.opponent_conditioned}", flush=True)
        train_route_ppo(
            e015_root=args.e015_root,
            out_path=model_path,
            episodes=spec.train_episodes,
            seed_start=6400000 + idx * 10000,
            opponent_conditioned=spec.opponent_conditioned,
            replay_prior=prior,
            reward_mode=spec.reward_mode,
            entropy_coef=spec.entropy_coef,
            episodes_per_update=16,
            workers=args.workers,
        )

    print("[E019] Round 1: 48 -> 12", flush=True)
    top12, _, _ = run_round(
        name="round1_screen", specs=specs,
        opponents=["e008a", "animal_fertuse", "crop_wide"],
        seeds=list(range(6100000, 6100002)), keep=12,
        e015_root=args.e015_root, e018_agent_dir=args.e018_agent_dir,
        replay_model_dir=family_dir, ppo_dir=models, results_dir=results, workers=args.workers,
    )

    print("[E019] Round 2: 12 -> 4", flush=True)
    top4, _, _ = run_round(
        name="round2_league", specs=top12,
        opponents=["e008a", "e007", "e018d1", "animal_fertuse", "dense_waterlock"],
        seeds=list(range(6200000, 6200004)), keep=4,
        e015_root=args.e015_root, e018_agent_dir=args.e018_agent_dir,
        replay_model_dir=family_dir, ppo_dir=models, results_dir=results, workers=args.workers,
    )

    # How: spend exact-environment compute only after the global search has narrowed 48 algorithms to four.
    if args.skip_official_gate:
        top3 = top4[:3]
        official_gate = {"skipped": True, "reason": "--skip-official-gate"}
    else:
        print("[E019] Official gate: 4 -> 3", flush=True)
        top3, official_games, official_board = run_official_gate(
            specs=top4,
            opponents=["e008a", "e007", "animal_fertuse", "crop_wide"],
            seeds=list(range(6500000, 6500004)),
            keep=3,
            e015_root=args.e015_root,
            e018_agent_dir=args.e018_agent_dir,
            replay_model_dir=family_dir,
            ppo_dir=models,
            results_dir=results,
            workers=min(args.official_workers, args.workers),
        )
        official_gate = {
            "skipped": False,
            "games": int(len(official_games)),
            "leaderboard": official_board.to_dict("records"),
        }

    submissions = []
    for spec in top3:
        archive = build_submission(
            spec=spec, e015_root=args.e015_root, e018_agent_dir=args.e018_agent_dir,
            replay_model_dir=family_dir, ppo_dir=models, source_dir=ROOT / "src", out_dir=out / "submissions",
        )
        submissions.append(str(archive))

    verdict = {
        "experiment_id": "E019",
        "search_mode": "global_successive_halving",
        "initial_population": len(specs),
        "family_summary": family_summary,
        "round1_top12": [s.candidate_id for s in top12],
        "round2_top4": [s.candidate_id for s in top4],
        "official_gate": official_gate,
        "final_top3": [s.candidate_id for s in top3],
        "top3_specs": [s.to_dict() for s in top3],
        "submission_archives": submissions,
        "protected_holdout_opened": False,
        "next": "submit promoted archives; use leaderboard/replay feedback to seed E019 generation-2 population",
    }
    (reports / "E019_VERDICT.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e015-root", type=Path, required=True)
    ap.add_argument("--e018-agent-dir", type=Path, required=True)
    ap.add_argument("--replay-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 2)))
    ap.add_argument("--official-workers", type=int, default=4)
    ap.add_argument("--refit-replay-families", action="store_true")
    ap.add_argument("--retrain-ppo", action="store_true")
    ap.add_argument("--skip-official-gate", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
