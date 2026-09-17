"""What: E019 always starts from the intended broad 48-arm population."""
from src.candidate_space import generate_candidate_specs


def test_candidate_population_is_48_unique_arms() -> None:
    # What: prevent accidental drift back to a small local parameter sweep.
    specs = generate_candidate_specs()
    assert len(specs) == 48
    assert len({s.candidate_id for s in specs}) == 48
    families = {s.family for s in specs}
    assert families == {"fixed", "phase_router", "heuristic_router", "replay_router", "route_ppo"}
    assert sum(s.family == "route_ppo" for s in specs) == 8
