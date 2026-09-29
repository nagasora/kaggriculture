"""What: E045が24手switch再発・非互換option選択・未観測Q学習を防ぐことを検証する。"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from kaggriculture_e045.losses import advantage_weighted_behavior_loss, masked_logits, option_value_loss
from kaggriculture_e045.macro import macro_boundary, plan_signature
from kaggriculture_e045.model import HierarchicalOptionTransformer
from kaggriculture_e045.schema import CounterfactualBatch
from kaggriculture_e045.selector import SafeOptionSelector, SelectorConfig


def test_macro_boundaries_are_three_days_not_24_turns():
    assert macro_boundary(0) and macro_boundary(72) and macro_boundary(144)
    assert not macro_boundary(24) and not macro_boundary(48) and not macro_boundary(73)


def test_selector_keeps_route_between_boundaries_and_after_day6():
    s = SafeOptionSelector(SelectorConfig(min_q_margin=0.01, route_switch_last_step=144))
    assert s.choose(0, [0.0, 1.0], [True, True]) == 1
    assert s.choose(24, [10.0, 0.0], [True, True]) == 1
    assert s.choose(216, [0.0, 100.0], [True, True]) == 0


def test_selector_masks_incompatible_and_low_margin_options():
    s = SafeOptionSelector(SelectorConfig(min_q_margin=0.2))
    assert s.choose(0, [0.5, 9.0, 0.6], [True, False, True]) == 0
    assert s.choose(72, [0.5, 9.0, 1.0], [True, False, True]) == 2


def test_plan_signature_counts_strategy_structure():
    actions = [{"farmer": ["PLANT", "WHEAT"], "hands": [["BUILD_PASTURE"]], "market": [["BUY_SEED", "WHEAT", 1], ["SELL", "MILK", 1]]} for _ in range(72)]
    sig = plan_signature(actions, 0)
    assert sig.plant_count == 72 and sig.build_count == 72
    assert sig.market_buy_count == 72 and sig.market_sell_count == 72


def test_transformer_scores_arbitrary_candidate_count():
    model = HierarchicalOptionTransformer(state_dim=7, option_dim=8, d_model=32, nhead=4, layers=1, max_history=6)
    out = model(torch.randn(3, 6, 7), torch.tensor([6, 4, 2]), torch.randn(3, 11, 8))
    assert out["q"].shape == (3, 11)
    assert out["value"].shape == (3,)
    assert out["residual_logits"].shape == (3, 4)


def test_masked_loss_never_uses_invalid_action():
    logits = torch.tensor([[1.0, 100.0, 2.0]])
    masked = masked_logits(logits, torch.tensor([[True, False, True]]))
    assert masked.argmax(1).item() == 2
    loss = advantage_weighted_behavior_loss(masked, torch.tensor([2]), torch.tensor([1.0]), torch.tensor([[True, False, True]]))
    assert torch.isfinite(loss)


def test_option_value_loss_only_observed_targets():
    q = torch.tensor([[1.0, 999.0]], requires_grad=True)
    target = torch.tensor([[0.0, -999.0]])
    mask = torch.tensor([[True, False]])
    loss = option_value_loss(q, target, mask)
    loss.backward()
    assert q.grad[0, 0] != 0 and q.grad[0, 1] == 0


def test_counterfactual_batch_requires_a_safe_option():
    batch = CounterfactualBatch(
        history=np.zeros((1, 2, 3), np.float32),
        valid_lengths=np.ones(1, np.int64),
        option_features=np.zeros((1, 2, 8), np.float32),
        option_returns=np.zeros((1, 2), np.float32),
        observed_mask=np.ones((1, 2), bool),
        compatible_mask=np.zeros((1, 2), bool),
        behavior_option=np.zeros(1, np.int64),
    )
    with pytest.raises(ValueError):
        batch.validate()
