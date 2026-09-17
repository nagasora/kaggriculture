"""PPO over macro expert routes for E019.

How:
- One action chooses one of eight proven macro experts for a six-day phase.
- Five decisions replace the original 720-turn primitive search, reducing credit-assignment depth.
- Optional replay priors initialize only the expert logits; PPO still learns from game outcomes.

Why not:
- Do not reuse E015's 108-way primitive macro grid here. E019 tests whether expert routing is the better abstraction.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

EXPERTS = (
    "dense", "early", "dense_waterlock", "early_waterlock",
    "crop_wide", "animal_core", "animal_fertsell", "animal_fertuse",
)
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("COW", "SHEEP", "GOOSE")


def encode_state(obs: dict, opponent_conditioned: bool = True) -> np.ndarray:
    p = int(obs["player"]); day = int(obs["step"]) // 24
    out: list[float] = [day / 29.0, (29 - day) / 29.0]
    for idx, seat in enumerate((p, 1 - p)):
        farm = obs["farms"][seat]; tiles = [t for row in farm["tiles"] for t in row]
        block = [
            float(farm["money"]) / 30000.0, len(farm["unlocked_quadrants"]) / 4.0,
            len(farm["hands"]) / 12.0,
            sum(isinstance(t, dict) and ("crop" in t or "animal" in t) for t in tiles) / 100.0,
            sum(isinstance(t, dict) and t.get("kind") == "WEED" for t in tiles) / 100.0,
        ]
        block.extend(sum(isinstance(t, dict) and t.get("crop") == crop for t in tiles) / 100.0 for crop in CROPS)
        block.extend(sum(isinstance(t, dict) and t.get("animal") == animal for t in tiles) / 12.0 for animal in ANIMALS)
        if idx == 1 and not opponent_conditioned: block = [0.0] * len(block)
        out.extend(block)
    market = obs["market"]
    for product in PRODUCTS:
        out.extend([float(market["prices"].get(product, 0.0)) / 250.0,
                    (float(market["inventory"].get(product, 10000.0)) - 10000.0) / 1000.0])
    out.append(len((obs.get("town") or {}).get("unlocked_shops", [])) / 12.0)
    shed = (obs.get("private") or {}).get("shed") or {}
    out.extend(float(shed.get(product, 0.0)) / 100.0 for product in PRODUCTS)
    x = np.asarray(out, dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError("nonfinite route state")
    return np.clip(x, -10, 10)


def valid_expert_mask(obs: dict) -> np.ndarray:
    """How: prevent switching back to crop-only legacy executors after livestock exists."""
    p = int(obs["player"]); tiles = [t for row in obs["farms"][p]["tiles"] for t in row]
    mask = np.ones(len(EXPERTS), dtype=bool)
    if any(isinstance(t, dict) and "animal" in t for t in tiles):
        mask[:] = False
        for name in ("animal_core", "animal_fertsell", "animal_fertuse"):
            mask[EXPERTS.index(name)] = True
    return mask


class NumpyRoutePolicy:
    def __init__(self, payload: dict, seed: int = 0):
        self.state = {k: np.asarray(v, dtype=np.float32) for k, v in payload["state"].items()}
        self.rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        w = self.state
        h = np.tanh(w["body.0.weight"] @ x + w["body.0.bias"])
        h = np.tanh(w["body.2.weight"] @ h + w["body.2.bias"])
        return w["actor.weight"] @ h + w["actor.bias"], float((w["critic.weight"] @ h + w["critic.bias"])[0])

    def choose(self, x: np.ndarray, deterministic: bool = True, mask: np.ndarray | None = None) -> tuple[int, float, float]:
        logits, value = self.forward(x); z = logits.astype(np.float64).copy()
        if mask is not None: z[~np.asarray(mask, dtype=bool)] = -1e9
        z -= float(np.max(z)); p = np.exp(z); p /= p.sum()
        action = int(np.argmax(p)) if deterministic else int(self.rng.choice(len(p), p=p))
        return action, float(math.log(max(p[action], 1e-12))), value


class RoutePolicyAgent:
    def __init__(self, ctx: Any, payload: dict, seed: int = 0, opponent_conditioned: bool = True, deterministic: bool = True):
        self.ctx = ctx; self.policy = NumpyRoutePolicy(payload, seed); self.seed = int(seed)
        self.opponent_conditioned = bool(opponent_conditioned); self.deterministic = bool(deterministic)
        self.phase, self.inner, self.route = -1, None, []

    def __call__(self, obs: dict, configuration: Any = None) -> dict:
        phase = min(4, int(obs["step"]) // 144)
        if phase != self.phase:
            self.phase = phase
            action, _, _ = self.policy.choose(encode_state(obs, self.opponent_conditioned), self.deterministic, valid_expert_mask(obs))
            name = EXPERTS[action]; self.route.append(name); self.inner = self.ctx.fixed(name, self.seed + phase * 2017)
        return self.inner(obs, configuration)


def _torch_model(n_features: int, prior: np.ndarray | None = None):
    import torch
    from torch import nn
    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(n_features, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh())
            self.actor = nn.Linear(96, len(EXPERTS)); self.critic = nn.Linear(96, 1)
            for m in self.modules():
                if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
            nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.critic.weight)
            if prior is not None:
                q = np.asarray(prior, dtype=np.float64); q = np.clip(q/q.sum(), 1e-4, 1.0)
                with torch.no_grad(): self.actor.bias.copy_(torch.tensor(np.log(q), dtype=torch.float32))
        def forward(self, x):
            h = self.body(x); return self.actor(h), self.critic(h).squeeze(-1)
    return ActorCritic()


def model_payload(model: Any) -> dict:
    return {"experts": list(EXPERTS), "state": {k: v.detach().cpu().numpy().tolist() for k,v in model.state_dict().items()}}


def replay_prior_from_profiles(profile_csv: Path) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(profile_csv); counts = np.ones(len(EXPERTS), dtype=np.float64) * 0.25
    for _, row in df.iterrows():
        expert = str(row.get("expert_prior", "crop_wide"))
        if expert in EXPERTS: counts[EXPERTS.index(expert)] += float(row.get("n_episodes", 1.0))
    return counts / counts.sum()


def _play_training_episode(*, e015_root: Path, payload: dict, seed: int, seat: int, opponent: str,
                           opponent_conditioned: bool, reward_mode: str) -> dict:
    root = str(e015_root)
    if root not in sys.path: sys.path.insert(0, root)
    from e014b2.simulator import ReplayValidatedSimulator, CONFIG  # type: ignore
    from e014b3.runner import initial_observations, make_agent  # type: ignore
    sim = ReplayValidatedSimulator(initial_observations(), int(seed)); opponent_agent = make_agent(opponent)
    pol = NumpyRoutePolicy(payload, int(seed)+5551)
    xs=[]; actions=[]; logps=[]; values=[]; rewards=[]; masks=[]; route=[]
    for phase, (start, stop) in enumerate(zip((0,144,288,432,576),(144,288,432,576,719))):
        if sim.step != start: raise RuntimeError((sim.step, start))
        obs = sim.obs(seat); x = encode_state(obs, opponent_conditioned); mask = valid_expert_mask(obs)
        action, lp, value = pol.choose(x, deterministic=False, mask=mask); expert_name = EXPERTS[action]; expert = make_agent(expert_name); route.append(expert_name)
        before = sim.farms[seat]["money"] - sim.farms[1-seat]["money"]
        while sim.step < stop:
            own = expert(sim.obs(seat), CONFIG); other = opponent_agent(sim.obs(1-seat), CONFIG)
            sim.advance([own, other] if seat == 0 else [other, own])
        after = sim.farms[seat]["money"] - sim.farms[1-seat]["money"]
        terminal_reward = float(np.sign(after)) if phase == 4 else 0.0
        shaped = (after-before)/20000.0 if reward_mode == "shaped" else 0.0
        xs.append(x); actions.append(action); logps.append(lp); values.append(value); rewards.append(terminal_reward+shaped); masks.append(mask)
    margin = float(sim.farms[seat]["money"] - sim.farms[1-seat]["money"])
    return {"x":np.stack(xs),"a":np.asarray(actions,np.int64),"lp":np.asarray(logps,np.float32),"v":np.asarray(values,np.float32),
            "r":np.asarray(rewards,np.float32),"mask":np.stack(masks).astype(bool),"margin":margin,
            "score":1.0 if margin>0 else (0.5 if margin==0 else 0.0),"route":route}


def _play_training_episode_kwargs(kwargs: dict) -> dict:
    return _play_training_episode(**kwargs)


def train_route_ppo(*, e015_root: Path, out_path: Path, episodes: int, seed_start: int,
                    opponent_conditioned: bool, replay_prior: np.ndarray | None, reward_mode: str = "terminal",
                    entropy_coef: float = 0.01, episodes_per_update: int = 16, workers: int = 8) -> dict:
    import torch
    from torch.distributions import Categorical
    root = str(e015_root)
    if root not in sys.path: sys.path.insert(0, root)
    from e014b3.runner import initial_observations  # type: ignore
    model = _torch_model(len(encode_state(initial_observations()[0], opponent_conditioned)), replay_prior)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    opponents = ("e008a","animal_fertuse","animal_fertsell","crop_wide","early_waterlock","dense_waterlock")
    history=[]; completed=0
    while completed < episodes:
        batch_n=min(episodes_per_update, episodes-completed); frozen=model_payload(model); jobs=[]
        for j in range(batch_n):
            ep=completed+j
            jobs.append(dict(e015_root=e015_root,payload=frozen,seed=seed_start+ep,seat=ep%2,opponent=opponents[ep%len(opponents)],
                             opponent_conditioned=opponent_conditioned,reward_mode=reward_mode))
        if workers>1:
            with ProcessPoolExecutor(max_workers=int(workers)) as pool: trajectories=list(pool.map(_play_training_episode_kwargs,jobs))
        else: trajectories=[_play_training_episode(**j) for j in jobs]
        xs=[]; aa=[]; oldlp=[]; adv=[]; ret=[]; masks=[]
        for tr in trajectories:
            r=tr["r"].astype(np.float64); v=tr["v"].astype(np.float64); gae=np.zeros_like(r); running=0.0
            for t in range(len(r)-1,-1,-1):
                next_v=0.0 if t==len(r)-1 else v[t+1]; delta=r[t]+next_v-v[t]; running=delta+0.95*running; gae[t]=running
            xs.append(tr["x"]); aa.append(tr["a"]); oldlp.append(tr["lp"]); adv.append(gae); ret.append(gae+v); masks.append(tr["mask"])
        x=torch.tensor(np.concatenate(xs),dtype=torch.float32); a=torch.tensor(np.concatenate(aa),dtype=torch.int64)
        old=torch.tensor(np.concatenate(oldlp),dtype=torch.float32); advantage=torch.tensor(np.concatenate(adv),dtype=torch.float32)
        returns=torch.tensor(np.concatenate(ret),dtype=torch.float32); valid_masks=torch.tensor(np.concatenate(masks),dtype=torch.bool)
        advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        for _ in range(4):
            logits,values=model(x); logits=logits.masked_fill(~valid_masks,-1e9); dist=Categorical(logits=logits); lp=dist.log_prob(a)
            ratio=torch.exp(lp-old); clipped=torch.clamp(ratio,0.8,1.2)*advantage
            loss=-torch.minimum(ratio*advantage,clipped).mean()+0.5*(values-returns).pow(2).mean()-float(entropy_coef)*dist.entropy().mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        completed += batch_n
        history.append({"episodes":completed,"mean_score":float(np.mean([t["score"] for t in trajectories])),"mean_margin":float(np.mean([t["margin"] for t in trajectories]))})
        print(f"[route-ppo] {out_path.stem}: {completed}/{episodes} score={history[-1]['mean_score']:.3f} margin={history[-1]['mean_margin']:+.0f}",flush=True)
    payload=model_payload(model); payload.update({"opponent_conditioned":bool(opponent_conditioned),"reward_mode":reward_mode,
        "entropy_coef":float(entropy_coef),"episodes":int(episodes),"seed_start":int(seed_start),"history":history})
    out_path.parent.mkdir(parents=True,exist_ok=True); out_path.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    return payload
