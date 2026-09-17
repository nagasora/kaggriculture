"""E019 robust ranking and successive-halving utilities.

How:
- Promote by win-oriented robustness across opponent families.
- Keep coin margin as a small tie-break diagnostic rather than the optimization target.

Why not:
- Do not promote a candidate only because it wins heavily against one weak opponent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_candidates(games: pd.DataFrame) -> pd.DataFrame:
    """How: aggregate mean/worst matchup score, seat-paired stability, and failures."""
    required = {"candidate_id", "opponent", "seed", "seat", "score", "margin", "status"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    g = games.copy()
    g["done"] = g["status"].eq("DONE")
    matchup = g.groupby(["candidate_id", "opponent"], as_index=False).agg(
        win_score=("score", "mean"),
        mean_margin=("margin", "mean"),
        done_rate=("done", "mean"),
    )
    paired = g.groupby(["candidate_id", "opponent", "seed"], as_index=False).agg(
        pair_score=("score", "mean"),
        pair_margin=("margin", "mean"),
    )

    rows = []
    for cid, part in matchup.groupby("candidate_id"):
        p = paired[paired.candidate_id.eq(cid)]
        avg_win = float(part.win_score.mean())
        worst_win = float(part.win_score.min())
        med_pair = float(p.pair_score.median())
        done_rate = float(part.done_rate.min())
        # Margin contribution is deliberately bounded and small.
        margin_bonus = float(np.tanh(p.pair_margin.median() / 20000.0))
        robust_score = (
            0.50 * avg_win
            + 0.30 * worst_win
            + 0.15 * med_pair
            + 0.05 * (0.5 + 0.5 * margin_bonus)
        )
        if done_rate < 1.0:
            robust_score -= 2.0 * (1.0 - done_rate)
        rows.append({
            "candidate_id": cid,
            "robust_score": robust_score,
            "avg_win_score": avg_win,
            "worst_matchup_score": worst_win,
            "median_pair_score": med_pair,
            "median_pair_margin": float(p.pair_margin.median()),
            "done_rate": done_rate,
        })
    return pd.DataFrame(rows).sort_values(
        ["robust_score", "worst_matchup_score", "avg_win_score"], ascending=False
    ).reset_index(drop=True)


def select_top(games: pd.DataFrame, n: int) -> list[str]:
    """How: return only the promoted IDs so each round can reuse the same evaluator."""
    board = summarize_candidates(games)
    return board.head(int(n)).candidate_id.tolist()
