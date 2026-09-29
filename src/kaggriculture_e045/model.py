"""How: 公開状態履歴と任意expert optionを同じ潜在空間で価値評価するTransformer。"""
from __future__ import annotations

import math
import torch
from torch import nn


class HierarchicalOptionTransformer(nn.Module):
    """How: state historyをcausal Transformerで符号化し、候補optionごとのQと残差modeを出す。"""

    def __init__(
        self,
        state_dim: int,
        option_dim: int,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 3,
        max_history: int = 16,
        residual_modes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_history = int(max_history)
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_history, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.state_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.option_encoder = nn.Sequential(
            nn.LayerNorm(option_dim), nn.Linear(option_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.q_bias = nn.Sequential(nn.LayerNorm(option_dim), nn.Linear(option_dim, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.residual_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, residual_modes))

    def encode_history(self, history: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        """How: paddingを除外し、未来attentionを禁止したlast-token表現を返す。"""
        if history.ndim != 3:
            raise ValueError("history must be [batch,time,state_dim]")
        if history.shape[1] > self.max_history:
            history = history[:, -self.max_history :]
        _, time, _ = history.shape
        valid_lengths = valid_lengths.clamp(min=1, max=time)
        h = self.state_proj(self.state_norm(history)) + self.position[:, :time]
        causal = torch.triu(torch.ones(time, time, dtype=torch.bool, device=history.device), diagonal=1)
        pos = torch.arange(time, device=history.device)[None, :]
        padding = pos < (time - valid_lengths[:, None])
        z = self.history_encoder(h, mask=causal, src_key_padding_mask=padding)
        return self.state_head(z[:, -1])

    def forward(
        self,
        history: torch.Tensor,
        valid_lengths: torch.Tensor,
        option_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """How: fixed classではなくcandidate-conditioned Qを出し、expert追加時の再学習依存を減らす。"""
        state = self.encode_history(history, valid_lengths)
        option = self.option_encoder(option_features)
        scale = math.sqrt(option.shape[-1])
        q = torch.einsum("bd,bkd->bk", state, option) / scale + self.q_bias(option_features).squeeze(-1)
        return {
            "q": q,
            "value": self.value_head(state).squeeze(-1),
            "residual_logits": self.residual_head(state),
            "state_embedding": state,
        }
