"""Massively parallel E019 arena with successive halving.

How:
- Round 1 screens all 48 algorithm families on a cheap reconstructed simulator.
- Later rounds spend more seeds/opponents only on promoted candidates.
- Every game swaps seats through the job matrix.

Why not:
- Do not run 32-seed confirmation for all 48 arms. That spends compute before locating a promising region.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import pandas as pd

from .agents import AssetContext, build_candidate
from .candidate_space import CandidateSpec
from .ranking import summarize_candidates

_CTX_CACHE: dict[tuple[str, str, str], AssetContext] = {}


def _ctx(e015_root: str, e018_agent_dir: str, replay_model_dir: str) -> AssetContext:
    key = (e015_root, e018_agent_dir, replay_model_dir)
    if key not in _CTX_CACHE:
        _CTX_CACHE[key] = AssetContext(Path(e015_root), Path(e018_agent_dir), Path(replay_model_dir))
    return _CTX_CACHE[key]


def _spec_from_dict(d: dict) -> CandidateSpec:
    x = dict(d)
    x["route"] = tuple(x.get("route") or ())
    x["phase_days"] = tuple(x.get("phase_days") or (0, 6, 12, 18, 24))
    return CandidateSpec(**x)


def run_match_job(job: dict) -> dict:
    """How: create fresh candidate/opponent instances and complete exactly one seed/seat match."""
    started = time.perf_counter()
    spec = _spec_from_dict(job["spec"])
    e015_root = Path(job["e015_root"])
    root = str(e015_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from e014b2.simulator import ReplayValidatedSimulator, CONFIG  # type: ignore
    from e014b3.runner import initial_observations  # type: ignore

    seed = int(job["seed"])
    seat = int(job["seat"])
    if 37000 <= seed <= 37031:
        raise ValueError("protected legacy holdout")
    ctx = _ctx(job["e015_root"], job["e018_agent_dir"], job["replay_model_dir"])
    payload = None
    ppo_path = job.get("ppo_path")
    if ppo_path:
        payload = json.loads(Path(ppo_path).read_text())

    try:
        candidate = build_candidate(spec, ctx, seed + seat * 1000003, payload)
        opponent = ctx.fixed(str(job["opponent"]), seed + 7000001 + seat)
        sim = ReplayValidatedSimulator(initial_observations(), seed)
        cand_time = 0.0
        opp_time = 0.0
        for _ in range(719):
            t = time.perf_counter(); own = candidate(sim.obs(seat), CONFIG); cand_time = max(cand_time, time.perf_counter() - t)
            t = time.perf_counter(); other = opponent(sim.obs(1 - seat), CONFIG); opp_time = max(opp_time, time.perf_counter() - t)
            actions = [own, other] if seat == 0 else [other, own]
            sim.advance(actions)
        mine = float(sim.farms[seat]["money"])
        theirs = float(sim.farms[1 - seat]["money"])
        margin = mine - theirs
        return {
            "candidate_id": spec.candidate_id, "family": spec.family, "opponent": str(job["opponent"]),
            "seed": seed, "seat": seat, "score": 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0),
            "margin": margin, "candidate_coin": mine, "opponent_coin": theirs, "status": "DONE",
            "elapsed_sec": time.perf_counter() - started, "max_candidate_call_sec": cand_time,
            "max_opponent_call_sec": opp_time, "error": "",
        }
    except Exception as exc:
        return {
            "candidate_id": spec.candidate_id, "family": spec.family, "opponent": str(job["opponent"]),
            "seed": seed, "seat": seat, "score": 0.0, "margin": -1e9,
            "candidate_coin": float("nan"), "opponent_coin": float("nan"), "status": "ERROR",
            "elapsed_sec": time.perf_counter() - started, "max_candidate_call_sec": float("nan"),
            "max_opponent_call_sec": float("nan"), "error": f"{type(exc).__name__}: {exc}",
        }


def make_jobs(specs: Iterable[CandidateSpec], opponents: Iterable[str], seeds: Iterable[int], *,
              e015_root: Path, e018_agent_dir: Path, replay_model_dir: Path, ppo_dir: Path) -> list[dict]:
    jobs = []
    for spec in specs:
        ppo = ppo_dir / f"{spec.candidate_id}.json"
        for opponent in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    jobs.append({
                        "spec": spec.to_dict(), "opponent": opponent, "seed": int(seed), "seat": seat,
                        "e015_root": str(e015_root), "e018_agent_dir": str(e018_agent_dir),
                        "replay_model_dir": str(replay_model_dir), "ppo_path": str(ppo) if ppo.exists() else "",
                    })
    return jobs


def run_jobs(jobs: list[dict], out_csv: Path, workers: int = 8) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    completed = set()
    if not done.empty:
        completed = set(zip(done.candidate_id, done.opponent, done.seed.astype(int), done.seat.astype(int)))
    pending = [j for j in jobs if (j["spec"]["candidate_id"], j["opponent"], int(j["seed"]), int(j["seat"])) not in completed]
    rows = [] if done.empty else done.to_dict("records")
    print(f"[arena] {out_csv.stem}: total={len(jobs)} cached={len(jobs)-len(pending)} pending={len(pending)} workers={workers}", flush=True)

    if workers <= 1:
        for i, job in enumerate(pending, 1):
            rows.append(run_match_job(job))
            if i % 10 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"[arena] {out_csv.stem}: {i}/{len(pending)} new games", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(run_match_job, j): j for j in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 20 == 0:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    print(f"[arena] {out_csv.stem}: {i}/{len(pending)} new games", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    return result


def run_round(*, name: str, specs: list[CandidateSpec], opponents: list[str], seeds: list[int], keep: int,
              e015_root: Path, e018_agent_dir: Path, replay_model_dir: Path, ppo_dir: Path,
              results_dir: Path, workers: int):
    games_path = results_dir / f"{name}_games.csv"
    jobs = make_jobs(specs, opponents, seeds, e015_root=e015_root, e018_agent_dir=e018_agent_dir,
                     replay_model_dir=replay_model_dir, ppo_dir=ppo_dir)
    games = run_jobs(jobs, games_path, workers=workers)
    board = summarize_candidates(games)
    board.to_csv(results_dir / f"{name}_leaderboard.csv", index=False)
    selected = set(board.head(int(keep)).candidate_id)
    promoted = [s for s in specs if s.candidate_id in selected]
    (results_dir / f"{name}_promoted.json").write_text(
        json.dumps([s.to_dict() for s in promoted], ensure_ascii=False, indent=2), encoding="utf-8")
    return promoted, games, board
