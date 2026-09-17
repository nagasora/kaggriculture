"""What: promotion rewards robust cross-opponent strength over one huge margin."""
import pandas as pd
from src.ranking import summarize_candidates


def test_robust_candidate_beats_spiky_margin_candidate() -> None:
    # What: a candidate that wins both matchups should outrank one that crushes only one matchup.
    rows = []
    for cid, scores, margins in [
        ("robust", [1, 1], [100, 100]),
        ("spiky", [1, 0], [100000, -100]),
    ]:
        for opponent, score, margin in zip(("a", "b"), scores, margins):
            rows.append(dict(candidate_id=cid, opponent=opponent, seed=1, seat=0, score=score, margin=margin, status="DONE"))
            rows.append(dict(candidate_id=cid, opponent=opponent, seed=1, seat=1, score=score, margin=margin, status="DONE"))
    board = summarize_candidates(pd.DataFrame(rows))
    assert board.iloc[0].candidate_id == "robust"
