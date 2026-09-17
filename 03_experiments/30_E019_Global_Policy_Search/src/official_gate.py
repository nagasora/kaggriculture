"""Official Kaggle-environments gate for only the E019 finalists.

How:
- Use the exact Kaggle environment after reconstructed-simulator global pruning.
- Swap both seats for every seed/opponent and rank with the same robust win-oriented score.

Why not:
- Do not spend official-environment compute on all 48 arms; reconstructed simulation is only a screening instrument.
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


def _field(x, name: str, default=None):
    return x.get(name, default) if isinstance(x, dict) else getattr(x, name, default)


def _spec_from_dict(d: dict) -> CandidateSpec:
    x = dict(d)
    x["route"] = tuple(x.get("route") or ())
    x["phase_days"] = tuple(x.get("phase_days") or (0, 6, 12, 18, 24))
    return CandidateSpec(**x)


def run_official_match_job(job: dict) -> dict:
    started = time.perf_counter()
    spec = _spec_from_dict(job["spec"])
    seed = int(job["seed"])
    seat = int(job["seat"])
    if 37000 <= seed <= 37031:
        raise ValueError("protected legacy holdout")

    root = str(job["e015_root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    from kaggle_environments import make  # type: ignore

    ctx = AssetContext(Path(job["e015_root"]), Path(job["e018_agent_dir"]), Path(job["replay_model_dir"]))
    ppo_path = Path(job["ppo_path"]) if job.get("ppo_path") else None
    payload = json.loads(ppo_path.read_text()) if ppo_path and ppo_path.exists() else None
    candidate = build_candidate(spec, ctx, seed + seat * 1000003, payload)
    opponent = ctx.fixed(str(job["opponent"]), seed + 7000001 + seat)
    call_times: list[float] = []

    def timed_candidate(obs, conf=None):
        t0 = time.perf_counter()
        action = candidate(obs, conf)
        call_times.append(time.perf_counter() - t0)
        return action

    agents = [timed_candidate, opponent] if seat == 0 else [opponent, timed_candidate]
    try:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        steps = env.run(agents)
        final = steps[-1]
        rewards = [float(_field(s, "reward", 0.0) or 0.0) for s in final]
        statuses = [str(_field(s, "status", "UNKNOWN")) for s in final]
        mine, theirs = rewards[seat], rewards[1 - seat]
        margin = mine - theirs
        done = statuses[seat] == "DONE" and statuses[1 - seat] == "DONE"
        return {
            "candidate_id": spec.candidate_id, "family": spec.family, "opponent": str(job["opponent"]),
            "seed": seed, "seat": seat,
            "score": 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0), "margin": margin,
            "candidate_coin": mine, "opponent_coin": theirs, "status": "DONE" if done else "ERROR",
            "candidate_status": statuses[seat], "opponent_status": statuses[1-seat],
            "elapsed_sec": time.perf_counter() - started,
            "max_candidate_call_sec": max(call_times) if call_times else float("nan"),
            "error": "" if done else f"candidate={statuses[seat]}, opponent={statuses[1-seat]}",
        }
    except Exception as exc:
        return {
            "candidate_id": spec.candidate_id, "family": spec.family, "opponent": str(job["opponent"]),
            "seed": seed, "seat": seat, "score": 0.0, "margin": -1e9,
            "candidate_coin": float("nan"), "opponent_coin": float("nan"), "status": "ERROR",
            "candidate_status": "ERROR", "opponent_status": "UNKNOWN",
            "elapsed_sec": time.perf_counter() - started,
            "max_candidate_call_sec": max(call_times) if call_times else float("nan"),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _jobs(specs: Iterable[CandidateSpec], opponents: Iterable[str], seeds: Iterable[int], *,
          e015_root: Path, e018_agent_dir: Path, replay_model_dir: Path, ppo_dir: Path) -> list[dict]:
    out = []
    for spec in specs:
        ppo = ppo_dir / f"{spec.candidate_id}.json"
        for opponent in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    out.append({
                        "spec": spec.to_dict(), "opponent": opponent, "seed": int(seed), "seat": seat,
                        "e015_root": str(e015_root), "e018_agent_dir": str(e018_agent_dir),
                        "replay_model_dir": str(replay_model_dir), "ppo_path": str(ppo) if ppo.exists() else "",
                    })
    return out


def run_official_gate(*, specs: list[CandidateSpec], opponents: list[str], seeds: list[int], keep: int,
                      e015_root: Path, e018_agent_dir: Path, replay_model_dir: Path, ppo_dir: Path,
                      results_dir: Path, workers: int = 4):
    out_csv = results_dir / "official_final_gate_games.csv"
    jobs = _jobs(specs, opponents, seeds, e015_root=e015_root, e018_agent_dir=e018_agent_dir,
                 replay_model_dir=replay_model_dir, ppo_dir=ppo_dir)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    completed = set()
    if not done.empty:
        completed = set(zip(done.candidate_id, done.opponent, done.seed.astype(int), done.seat.astype(int)))
    pending = [j for j in jobs if (j["spec"]["candidate_id"], j["opponent"], int(j["seed"]), int(j["seat"])) not in completed]
    rows = [] if done.empty else done.to_dict("records")
    print(f"[official] total={len(jobs)} cached={len(jobs)-len(pending)} pending={len(pending)} workers={workers}", flush=True)

    if workers <= 1:
        for i, job in enumerate(pending, 1):
            rows.append(run_official_match_job(job))
            if i % 4 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"[official] {i}/{len(pending)} new games", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_official_match_job, j): j for j in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 4 == 0:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    print(f"[official] {i}/{len(pending)} new games", flush=True)

    games = pd.DataFrame(rows)
    games.to_csv(out_csv, index=False)
    board = summarize_candidates(games)
    board.to_csv(results_dir / "official_final_gate_leaderboard.csv", index=False)
    selected = set(board.head(int(keep)).candidate_id)
    promoted = [s for s in specs if s.candidate_id in selected]
    (results_dir / "official_final_gate_promoted.json").write_text(
        json.dumps([s.to_dict() for s in promoted], ensure_ascii=False, indent=2), encoding="utf-8")
    return promoted, games, board
