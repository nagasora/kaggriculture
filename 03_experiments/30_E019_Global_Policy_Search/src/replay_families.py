"""Top-replay strategy-family discovery for E019.

How:
- Aggregate each winning episode-seat into trajectory features at days 0/6/12/18/24/29.
- Fit family clusters only on train-split winning trajectories.
- Assign validation/test trajectories by nearest train centroid.

Why not:
- Do not cluster primitive 720-turn actions; that mostly learns timing noise rather than macro strategy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DAYS = (0, 6, 12, 18, 24, 29)
STATE_VARS = (
    "money", "opp_money", "money_margin", "unlocked_quadrants", "productive_tiles", "total_stock", "shop_count",
    "opp_unlocked_quadrants", "opp_productive_tiles",
    "crop_WHEAT", "crop_CARROT", "crop_TOMATO", "crop_STRAWBERRY", "crop_MELON",
    "animal_COW", "animal_SHEEP", "animal_GOOSE",
)
ACTION_PREFIXES = (
    "action_hire_count", "action_buy_land_count", "action_market_order_count",
    "action_buy_seed_", "action_buy_animal_", "action_sell_attempt_",
)


def _action_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if any(c == p or c.startswith(p) for p in ACTION_PREFIXES)]


def build_episode_features(daily: pd.DataFrame) -> pd.DataFrame:
    win = daily[daily["episode_result"].eq("WIN")].copy()
    key = ["episode_id", "seat"]
    meta = win[key + ["dataset_split", "team_name", "rank_at_acquisition", "sample_weight"]].drop_duplicates(key)
    actions = _action_columns(win.columns)
    out = meta.merge(win.groupby(key)[actions].sum().add_prefix("sum_").reset_index(), on=key, how="left")
    for day in DAYS:
        part = win[win.day.eq(day)][key + list(STATE_VARS)].copy()
        part = part.rename(columns={c: f"d{day}_{c}" for c in STATE_VARS})
        out = out.merge(part, on=key, how="left")
    return out.fillna(0)


def _family_name(profile: pd.Series) -> str:
    animals = sum(float(profile.get(f"d29_animal_{a}", 0.0)) for a in ("COW", "SHEEP", "GOOSE"))
    q6 = float(profile.get("d6_unlocked_quadrants", 1.0))
    prod29 = float(profile.get("d29_productive_tiles", 0.0))
    stock29 = float(profile.get("d29_total_stock", 0.0))
    if q6 >= 1.5 and animals >= 8: return "early-expand-livestock"
    if q6 >= 1.5: return "early-expand-mixed"
    if animals >= 14: return "livestock-heavy"
    if animals >= 8 and stock29 >= 80: return "livestock-stock"
    if animals >= 8: return "livestock-balanced"
    if prod29 >= 55: return "crop-persistent"
    if stock29 >= 85: return "crop-stock"
    return "crop-liquidating"


def _family_expert(profile: pd.Series) -> str:
    animals = sum(float(profile.get(f"d29_animal_{a}", 0.0)) for a in ("COW", "SHEEP", "GOOSE"))
    q6 = float(profile.get("d6_unlocked_quadrants", 1.0))
    fert_sell = float(profile.get("sum_action_sell_attempt_FERTILIZER", 0.0))
    if animals >= 8 and fert_sell >= 220: return "animal_fertsell"
    if animals >= 8: return "animal_fertuse"
    if q6 >= 1.5: return "early_waterlock"
    return "crop_wide"


def fit_families(csv_path: Path, out_dir: Path, candidate_ks: tuple[int, ...] = (6, 8, 10)) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(csv_path)
    episodes = build_episode_features(daily)
    meta = {"episode_id", "seat", "dataset_split", "team_name", "rank_at_acquisition", "sample_weight"}
    features = [c for c in episodes.columns if c not in meta]
    train_mask = episodes.dataset_split.eq("train")
    scaler = StandardScaler().fit(episodes.loc[train_mask, features])
    x_train = scaler.transform(episodes.loc[train_mask, features])

    silhouette: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=19017, n_init=20).fit(x_train)
        silhouette[k] = float(silhouette_score(x_train, model.labels_, sample_size=min(2500, len(x_train)), random_state=19017))
        models[k] = model
    best_k = max(silhouette, key=silhouette.get)
    model = models[best_k]
    episodes["family_id"] = model.predict(scaler.transform(episodes[features]))

    profiles = episodes.groupby("family_id")[features].median()
    counts = episodes.groupby("family_id").size().rename("n_episodes")
    profiles = profiles.join(counts)
    profiles["family_name"] = [f"F{i:02d}-{_family_name(row)}" for i, row in profiles.iterrows()]
    profiles["expert_prior"] = [_family_expert(row) for _, row in profiles.iterrows()]
    episodes["family_name"] = episodes.family_id.map(profiles.family_name.to_dict())

    profiles.reset_index().to_csv(out_dir / "family_profiles.csv", index=False)
    episodes.to_csv(out_dir / "episode_family.csv", index=False)
    labeled_daily = daily.merge(episodes[["episode_id", "seat", "family_id"]], on=["episode_id", "seat"], how="inner")
    day_profiles = labeled_daily.groupby(["family_id", "day"])[list(STATE_VARS)].median().reset_index()
    day_profiles["expert_prior"] = day_profiles.family_id.map(profiles.expert_prior.to_dict())
    day_profiles.to_csv(out_dir / "day_family_profiles.csv", index=False)
    joblib.dump({"scaler": scaler, "kmeans": model, "features": features, "profiles": profiles}, out_dir / "replay_families.joblib", compress=3)
    summary = {
        "n_daily_rows": int(len(daily)), "n_winning_episode_seats": int(len(episodes)), "best_k": int(best_k),
        "silhouette": {str(k): v for k, v in silhouette.items()},
        "family_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "family_expert_prior": {str(int(k)): str(v) for k, v in profiles.expert_prior.items()},
    }
    (out_dir / "family_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
