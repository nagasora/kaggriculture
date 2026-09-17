"""What: route-level action masking preserves executor state contracts."""
from __future__ import annotations

import numpy as np
from src.route_ppo import EXPERTS, valid_expert_mask


def _obs(tile):
    farm = {"money": 1000, "unlocked_quadrants": [0], "hands": [], "tiles": [[tile]]}
    return {"player": 0, "farms": [farm, farm]}


def test_all_experts_are_available_before_livestock() -> None:
    # What: crop-only states keep the full global route search space open.
    mask = valid_expert_mask(_obs({"crop": "WHEAT"}))
    assert mask.dtype == np.bool_
    assert mask.all()


def test_livestock_state_masks_crop_only_experts() -> None:
    # What: once livestock exists, only livestock-compatible executors remain legal.
    mask = valid_expert_mask(_obs({"animal": "COW"}))
    legal = {EXPERTS[i] for i, ok in enumerate(mask) if ok}
    assert legal == {"animal_core", "animal_fertsell", "animal_fertuse"}
