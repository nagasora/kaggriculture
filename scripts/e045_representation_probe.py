"""How: E037因果特徴でcurrent-state MLPとcausal history Transformerをseed分離比較する。"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    history_tokens: int = 10
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 20260929


class MacroHistoryDataset(Dataset):
    """How: macro境界までの過去だけを切り出し、future leakageを防ぐ。"""

    def __init__(self, X: np.ndarray, meta: list[dict], split: str, history_tokens: int):
        families = sorted({m["team"] for m in meta})
        self.idx_to_family = families
        family_to_idx = {f: i for i, f in enumerate(families)}
        self.samples = []
        for seat_idx, m in enumerate(meta):
            if m["split"] != split:
                continue
            for block in range(10):
                boundary_idx = min(block * 9, X.shape[1] - 1)
                lo = max(0, boundary_idx - history_tokens + 1)
                hist = X[seat_idx, lo : boundary_idx + 1]
                pad = np.zeros((history_tokens, X.shape[2]), dtype=np.float32)
                pad[-len(hist) :] = hist
                self.samples.append((pad, family_to_idx[m["team"]], block, len(hist)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, block, valid = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y), torch.tensor(block), torch.tensor(valid)


class CurrentMLP(nn.Module):
    """How: macro境界の現在状態だけでfamilyを分類する対照モデル。"""

    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1])


class CausalMacroTransformer(nn.Module):
    """How: 3日までの公開状態履歴をcausal attentionで圧縮してfamilyを推定する。"""

    def __init__(self, in_dim: int, classes: int, cfg: Config):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.history_tokens, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, classes))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(x)) + self.pos[:, : x.shape[1]]
        time = x.shape[1]
        # Why not: full attentionは未来観測を漏らすので使用しない。
        causal = torch.triu(torch.ones(time, time, device=x.device, dtype=torch.bool), diagonal=1)
        positions = torch.arange(time, device=x.device)[None, :]
        pad = positions < (time - valid[:, None])
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad)
        return self.head(h[:, -1])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, ds: Dataset, device: str) -> dict:
    model.eval()
    correct = total = 0
    block_c = {i: [0, 0] for i in range(10)}
    for x, y, block, valid in DataLoader(ds, batch_size=256):
        pred = model(x.to(device), valid.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
        for yy, pp, bb in zip(y.tolist(), pred.tolist(), block.tolist()):
            block_c[bb][0] += int(yy == pp)
            block_c[bb][1] += 1
    return {"accuracy": correct / total, "by_block": {str(k): a / n for k, (a, n) in block_c.items()}}


def fit(model: nn.Module, train: Dataset, test: Dataset, cfg: Config, device: str) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    best = None
    for epoch in range(cfg.epochs):
        model.train()
        for x, y, _, valid in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            loss = nn.functional.cross_entropy(model(x, valid), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        metrics = evaluate(model, test, device)
        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = {**metrics, "epoch": epoch + 1}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    cfg = Config(epochs=args.epochs)
    seed_all(cfg.seed)
    z = np.load(args.npz)
    X = z["X"].astype(np.float32)
    meta = json.loads(args.manifest.read_text())
    train_seat = np.array([m["split"] == "train" for m in meta])
    mu = X[train_seat].reshape(-1, X.shape[-1]).mean(0)
    sd = X[train_seat].reshape(-1, X.shape[-1]).std(0)
    sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd
    train = MacroHistoryDataset(X, meta, "train", cfg.history_tokens)
    test = MacroHistoryDataset(X, meta, "test", cfg.history_tokens)
    classes = len(train.idx_to_family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = fit(CurrentMLP(X.shape[-1], classes), train, test, cfg, device)
    seed_all(cfg.seed)
    transformer = fit(CausalMacroTransformer(X.shape[-1], classes, cfg), train, test, cfg, device)
    result = {
        "config": cfg.__dict__,
        "device": device,
        "train_samples": len(train),
        "test_samples": len(test),
        "families": train.idx_to_family,
        "future_inputs": False,
        "current_mlp": mlp,
        "causal_transformer": transformer,
        "delta_accuracy": transformer["accuracy"] - mlp["accuracy"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
