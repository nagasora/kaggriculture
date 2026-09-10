"""Replay の実測 coin 時系列と API の勝敗を照合する。"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .storage import read_csv, read_json, validate_replay, write_csv, write_json

EXPERIMENT = re.compile(r"(?<![A-Z0-9])E\d{3}[A-Z]?(?![A-Z0-9])", re.I)


def number(value: object) -> float | None:
    """How: 未取得と0を区別し、非有限数を除外する。"""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def experiment_id(submission: dict, overrides: dict) -> str:
    """How: 明示対応表、ファイル名・説明の順に E003A/B まで識別する。"""
    sid = str(submission.get("ref", ""))
    if sid in overrides:
        return str(overrides[sid])
    text = f"{submission.get('fileName', '')} {submission.get('description', '')}"
    found = set(EXPERIMENT.findall(text.upper()))
    return next(iter(found)) if len(found) == 1 else "UNMAPPED"


def outcome(my_reward: object, opponent_reward: object, completed: bool) -> str:
    """How: 完了対戦の既知 reward だけを比較する。"""
    mine, theirs = number(my_reward), number(opponent_reward)
    if not completed or mine is None or theirs is None:
        return "UNKNOWN"
    return "WIN" if mine > theirs else "LOSS" if mine < theirs else "DRAW"


def replay_metrics(data: dict, seat: int) -> tuple[dict, list[dict], list[dict]]:
    """How: seat で farm を選び、同じ観測の money 同士を比較する。"""
    if seat not in (0, 1):
        raise ValueError("seat は0か1である必要があります")
    curve = []
    days = {}
    own_ops = Counter()
    for step, frame in enumerate(data["steps"]):
        if len(frame) != 2:
            raise ValueError("二人対戦ではありません")
        obs = frame[seat].get("observation") or {}
        farms = obs.get("farms")
        if not isinstance(farms, list) or len(farms) != 2:
            continue
        mine, theirs = number(farms[seat].get("money")), number(farms[1-seat].get("money"))
        if mine is None or theirs is None:
            continue
        day, hour = obs.get("day"), obs.get("hour")
        row = {"step": step, "day": day, "hour": hour, "my_coins": mine,
               "opponent_coins": theirs, "coin_margin": mine - theirs}
        curve.append(row)
        if isinstance(day, int):
            days[day] = row
        action = frame[seat].get("action") or {}
        if step and isinstance(action, dict):
            farmer = action.get("farmer") or []
            if isinstance(farmer, list) and farmer:
                own_ops[str(farmer[0])] += 1
    final_frame = data["steps"][-1]
    final_obs = final_frame[seat].get("observation") or {}
    reward = final_frame[seat].get("reward")
    other_reward = final_frame[1-seat].get("reward")
    result = outcome(reward, other_reward, True)
    metrics = {"replay_status": "valid", "my_status": final_frame[seat].get("status"),
               "opponent_status": final_frame[1-seat].get("status"),
               "replay_reward": reward, "replay_opponent_reward": other_reward,
               "replay_outcome": result, "steps": len(data["steps"]),
               "module_version": data.get("module_version"), "seed": data.get("info", {}).get("seed"),
               "shops": final_obs.get("town", {}).get("unlocked_shops", []),
               "farmer_action_counts": dict(own_ops),
               "unsold_shed": final_obs.get("private", {}).get("shed", {}),
               "final_market_prices": final_obs.get("market", {}).get("prices", {})}
    if not curve:
        return metrics, curve, []
    # Why not the last observed money: 異常終了時の古い観測を最終値と呼ばないため。
    if curve[-1]["step"] == len(data["steps"]) - 1:
        metrics.update(my_final_coins=curve[-1]["my_coins"], opponent_final_coins=curve[-1]["opponent_coins"],
                       coin_margin=curve[-1]["coin_margin"])
    metrics["max_coin_lead"] = max(row["coin_margin"] for row in curve)
    leads = [r for r in curve if r["coin_margin"] > 0]
    metrics["last_lead_step"] = leads[-1]["step"] if leads else None
    metrics["last_lead_day"] = leads[-1]["day"] if leads else None
    last_lead = metrics["last_lead_step"]
    after = [r for r in curve if last_lead is not None and r["step"] > last_lead and r["coin_margin"] < 0]
    metrics["decisive_reversal_step"] = after[0]["step"] if result == "LOSS" and after else None
    metrics["decisive_reversal_day"] = after[0]["day"] if result == "LOSS" and after else None
    for checkpoint in (6, 24, 27):
        exact = next((r for r in curve if r["day"] == checkpoint and r["hour"] == 0), None)
        margin = exact["coin_margin"] if exact else None
        metrics[f"coin_margin_day{checkpoint}"] = margin
        if checkpoint != 6:
            metrics[f"lead_day{checkpoint}_then_loss"] = (margin > 0 and result == "LOSS") if margin is not None else None
    return metrics, curve, [days[d] for d in sorted(days)]


def summarize(rows: list[dict], key: str) -> list[dict]:
    """How: 未知勝敗を分母から外し、逆転率の分母を明記する。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "UNKNOWN")].append(row)
    result = []
    for label, group in sorted(groups.items(), key=lambda x: str(x[0])):
        counts = Counter(r["outcome"] for r in group)
        known = sum(counts[k] for k in ("WIN", "LOSS", "DRAW"))
        item = {key: label, "matches": len(group), **{k: counts[k] for k in ("WIN", "LOSS", "DRAW", "UNKNOWN")},
                "win_rate": counts["WIN"] / known if known else None,
                "valid_replays": sum(r.get("replay_status") == "valid" for r in group)}
        margins = [number(r.get("coin_margin")) for r in group]
        margins = [x for x in margins if x is not None]
        item["mean_coin_margin"] = mean(margins) if margins else None
        for day in (24, 27):
            evaluated = [r for r in group if r.get("replay_status") == "valid" and r["outcome"] != "UNKNOWN"
                         and number(r.get(f"coin_margin_day{day}")) is not None]
            leaders = [r for r in evaluated if r[f"coin_margin_day{day}"] > 0]
            reversed_ = [r for r in leaders if r["outcome"] == "LOSS"]
            item.update({f"evaluated_day{day}": len(evaluated), f"leads_day{day}": len(leaders),
                         f"reversal_losses_day{day}": len(reversed_),
                         f"reversal_rate_among_leads_day{day}": len(reversed_) / len(leaders) if leaders else None})
        result.append(item)
    return result


def analyze(root: Path, mapping_path: Path | None = None) -> dict:
    """How: API と Replay を照合し、master・日別・step別・実験別表を出す。"""
    submissions = read_json(root / "submissions.json", None)
    if submissions is None:
        submissions = read_csv(root / "submissions.csv")
    overrides = read_json(mapping_path, {}) if mapping_path else {}
    submission_map = {str(r.get("ref")): r for r in submissions}
    indexed = read_csv(root / "match_index.csv")
    by_episode = defaultdict(list)
    for row in indexed:
        by_episode[row.get("episode_id") or row.get("episodeId") or row.get("id")].append(row)
    masters, curves, daily = [], [], []
    errors = []
    valid_files = 0
    for eid, matches in sorted(by_episode.items(), key=lambda x: str(x[0])):
        data = None
        try:
            data = validate_replay(root / "replays" / f"{eid}.json")
            valid_files += 1
        except (ValueError, OSError, TypeError):
            pass
        for raw in matches:
            sid = str(raw["submission_id"])
            sub = submission_map.get(sid, {})
            tags = {"experiment_id": experiment_id(sub, overrides), "submission_id": sid, "episode_id": eid}
            row = {**raw, **tags, "public_score_snapshot": sub.get("publicScore"),
                   "replay_status": "missing_or_invalid"}
            row["competitive"] = raw.get("episode_type") == "EPISODE_TYPE_PUBLIC" and str(raw.get("seat")) in ("0", "1")
            row["outcome"] = outcome(raw.get("reward"), raw.get("opponent_reward"), raw.get("episode_state") == "COMPLETED")
            if data is not None and str(raw.get("seat")) in ("0", "1"):
                try:
                    reported = data.get("info", {}).get("EpisodeId")
                    if reported is not None and str(reported) != str(eid):
                        raise ValueError("EpisodeId 不一致")
                    metrics, curve, day_rows = replay_metrics(data, int(raw["seat"]))
                    row.update(metrics)
                    api_rewards = [number(raw.get("reward")), number(raw.get("opponent_reward"))]
                    replay_rewards = [number(metrics.get("replay_reward")), number(metrics.get("replay_opponent_reward"))]
                    disagreement = any(a is not None and b is not None and a != b for a, b in zip(api_rewards, replay_rewards))
                    if disagreement:
                        row.update(outcome="UNKNOWN", replay_status="reward_mismatch")
                        errors.append({**tags, "error": "API と Replay の reward 不一致"})
                    elif row["outcome"] == "UNKNOWN" and raw.get("episode_state") == "COMPLETED":
                        row["outcome"] = metrics["replay_outcome"]
                    curves.extend({**tags, **r} for r in curve)
                    daily.extend({**tags, **r} for r in day_rows)
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    row["replay_status"] = "schema_error"
                    errors.append({**tags, "error": type(exc).__name__})
            masters.append(row)
        del data
    out = root / "analysis"
    write_csv(out / "match_master.csv", masters, ["experiment_id", "submission_id", "episode_id", "outcome"])
    write_csv(out / "coin_timeseries.csv", curves, ["experiment_id", "submission_id", "episode_id", "step", "day", "hour", "coin_margin"])
    write_csv(out / "daily_coin_margin.csv", daily, ["experiment_id", "submission_id", "episode_id", "day", "coin_margin"])
    competitive = [r for r in masters if r["competitive"]]
    experiments = summarize(competitive, "experiment_id")
    write_csv(out / "experiment_summary.csv", experiments, ["experiment_id", "matches", "WIN", "LOSS", "DRAW", "UNKNOWN", "win_rate"])
    write_json(out / "experiment_summary.json", experiments)
    write_csv(out / "submission_summary.csv", summarize(competitive, "submission_id"), ["submission_id", "matches", "win_rate"])
    seats = [{**r, "experiment_seat": f"{r['experiment_id']}/seat{r.get('seat')}"} for r in competitive]
    opponents = [{**r, "experiment_opponent": f"{r['experiment_id']}/{r.get('opponent')}"} for r in competitive]
    write_csv(out / "seat_summary.csv", summarize(seats, "experiment_seat"), ["experiment_seat", "matches", "win_rate"])
    write_csv(out / "opponent_summary.csv", summarize(opponents, "experiment_opponent"), ["experiment_opponent", "matches", "win_rate"])
    failures = [r for r in masters if r["outcome"] in ("LOSS", "DRAW")]
    write_csv(out / "failure_index.csv", failures, ["experiment_id", "episode_id", "outcome", "replay_path"])
    audit = {"matches": len(masters), "unique_episodes": len(by_episode), "errors": errors,
             "valid_replay_files": valid_files, "competitive_matches": len(competitive),
             "excluded_from_competitive_summary": len(masters) - len(competitive),
             "unknown_competitive_outcomes": sum(r["outcome"] == "UNKNOWN" for r in competitive),
             "valid_replays": sum(r.get("replay_status") == "valid" for r in masters),
             "unknown_outcomes": sum(r["outcome"] == "UNKNOWN" for r in masters)}
    write_json(out / "analysis_audit.json", audit)
    return {"audit": audit, "experiments": experiments}
