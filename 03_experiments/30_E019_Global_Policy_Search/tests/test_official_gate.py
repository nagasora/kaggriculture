"""What: the final official gate cannot accidentally reopen the legacy closed holdout."""
from __future__ import annotations

import pytest
from src.candidate_space import generate_candidate_specs
from src.official_gate import run_official_match_job


def test_protected_legacy_holdout_is_rejected_before_environment_import() -> None:
    # What: seeds 37000-37031 remain closed even when kaggle-environments is unavailable locally.
    spec = generate_candidate_specs()[0]
    with pytest.raises(ValueError, match="protected legacy holdout"):
        run_official_match_job({
            "spec": spec.to_dict(), "opponent": "e008a", "seed": 37000, "seat": 0,
            "e015_root": "/missing", "e018_agent_dir": "/missing", "replay_model_dir": "/missing", "ppo_path": "",
        })
