"""How: imitationを壊さずcounterfactual returnでexpert選択を改善する損失を定義する。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """How: 工程非互換optionへ確率質量が流れないようhard maskする。"""
    if logits.shape != valid_mask.shape:
        raise ValueError("mask shape mismatch")
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("each sample needs at least one valid option")
    return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)


def advantage_weighted_behavior_loss(
    q_logits: torch.Tensor,
    behavior_option: torch.Tensor,
    advantage: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 0.5,
    max_weight: float = 20.0,
) -> torch.Tensor:
    """How: 良いteacher行動ほど強く模倣し、negative/low-advantage例の支配を防ぐ。"""
    logits = masked_logits(q_logits, valid_mask)
    weight = torch.exp(advantage / max(float(temperature), 1e-6)).clamp(max=float(max_weight)).detach()
    ce = F.cross_entropy(logits, behavior_option, reduction="none")
    return (weight * ce).mean()


def option_value_loss(
    q_values: torch.Tensor,
    option_returns: torch.Tensor,
    observed_mask: torch.Tensor,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """How: simulatorで実測した候補だけでQを回帰し、未評価optionを0ラベルにしない。"""
    if q_values.shape != option_returns.shape or q_values.shape != observed_mask.shape:
        raise ValueError("value target shape mismatch")
    if not observed_mask.any():
        raise ValueError("at least one observed option is required")
    return F.huber_loss(q_values[observed_mask], option_returns[observed_mask], delta=huber_delta)
