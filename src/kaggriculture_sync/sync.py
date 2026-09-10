"""中断しても再実行できる差分同期。"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from .storage import read_csv, read_json, replay_valid, validate_replay, write_csv, write_json


class Source(Protocol):
    """同期元の最小インターフェース。"""
    def submission_page(self, competition: str, token: str) -> tuple[list[dict], str]: ...
    def episodes(self, submission_id: str) -> list[dict]: ...
    def replay(self, episode_id: str, destination: Path) -> None: ...


def identifier(value: object) -> str:
    """How: API ID を正整数に限定してファイルパスを構築する。"""
    text = str(value)
    if not text.isascii() or not text.isdigit() or int(text) <= 0:
        raise ValueError("API ID が正整数ではありません")
    return str(int(text))


def all_submissions(source: Source, competition: str) -> list[dict]:
    """How: 終端トークンまで巡回し、ページ間重複を取り除く。"""
    found: dict[str, dict] = {}
    token = ""
    seen = set()
    while True:
        rows, next_token = source.submission_page(competition, token)
        for row in rows:
            found[identifier(row["ref"])] = row
        if not next_token:
            return list(found.values())
        if next_token in seen or next_token == token:
            raise ValueError("提出ページのトークンが循環しています")
        seen.add(next_token)
        token = next_token


def match_row(submission_id: str, episode: dict) -> dict:
    """How: episode と submission の組を1行にし、自分の seat を照合する。"""
    agents = episode.get("agents") or []
    own = [a for a in agents if str(a.get("submissionId")) == submission_id]
    me = own[0] if len(own) == 1 else {}
    opponents = [a for a in agents if a is not me]
    opponent = opponents[0] if len(opponents) == 1 else {}
    # Why not array position: agents 配列が seat 順であるとは限らないため。
    return {
        "submission_id": submission_id, "episode_id": identifier(episode["id"]),
        "episode_state": episode.get("state"), "episode_type": episode.get("type"),
        "seat_mapping": "unique" if len(own) == 1 else "self_play" if len(own) > 1 else "missing",
        "create_time": episode.get("createTime"), "end_time": episode.get("endTime"),
        "seat": me.get("index", 0) if me else None,
        "reward": me.get("reward"), "agent_state": me.get("state"),
        "opponent": opponent.get("teamName"), "opponent_submission_id": opponent.get("submissionId"),
        "opponent_reward": opponent.get("reward"), "opponent_agent_state": opponent.get("state"),
        "agents": agents, "replay_path": f"replays/{episode['id']}.json",
    }


def safe_error(exc: Exception) -> str:
    """How: 認証ヘッダーや署名 URL を含めず、種別と HTTP 番号だけ残す。"""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return type(exc).__name__ + (f" (HTTP {code})" if code is not None else "")


def sync(source: Source, output: Path, competition: str = "kaggriculture", workers: int = 3,
         max_replays: int | None = None, progress: Callable[[str], None] = print) -> dict:
    """How: 索引を先に保存し、Replay ごとに検証後のファイルを確定する。"""
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    state = {"started_at": started, "competition": competition, "status": "running", "errors": []}
    write_json(output / "sync_state.json", state)
    try:
        submissions = all_submissions(source, competition)
    except Exception as exc:
        state.update(status="failed", errors=[{"stage": "submissions", "error": safe_error(exc)}])
        write_json(output / "sync_state.json", state)
        raise
    previous = read_json(output / "submissions.json", [])
    merged = {identifier(row["ref"]): row for row in previous}
    merged.update({identifier(row["ref"]): row for row in submissions})
    write_json(output / "submissions.json", list(merged.values()))
    write_csv(output / "submissions.csv", list(merged.values()), ["ref", "fileName", "publicScore"])
    snapshot = output / "snapshots" / started.replace(":", "-")
    write_json(snapshot / "submissions.json", submissions)

    matches: dict[tuple[str, str], dict] = {}
    for row in read_csv(output / "match_index.csv"):
        sid = row.get("submission_id")
        eid = row.get("episode_id") or row.get("episodeId") or row.get("id")
        if sid and eid and str(eid).isdigit():
            row.update(submission_id=identifier(sid), episode_id=identifier(eid))
            matches[(row["submission_id"], row["episode_id"])] = row
    for sid in merged:
        episode_path = output / "episodes" / f"{sid}.json"
        old = read_json(episode_path, [])
        try:
            episodes = source.episodes(sid)
            write_json(snapshot / f"episodes_{sid}.json", episodes)
        except Exception as exc:
            state["errors"].append({"stage": "episodes", "submission_id": sid, "error": safe_error(exc)})
            episodes = []
        union = {identifier(row["id"]): row for row in old}
        union.update({identifier(row["id"]): row for row in episodes})
        write_json(episode_path, list(union.values()))
        write_csv(output / "episodes" / f"{sid}.csv", list(union.values()), ["id", "state", "agents"])
        for episode in union.values():
            row = match_row(sid, episode)
            matches[(sid, row["episode_id"])] = row
        progress(f"submission {sid}: {len(episodes)} returned, {len(union)} archived episodes")
        write_json(output / "sync_state.json", state)

    rows = sorted(matches.values(), key=lambda r: (int(r["submission_id"]), int(r["episode_id"])))
    write_csv(output / "match_index.csv", rows, ["submission_id", "episode_id", "seat", "reward"])
    eligible = sorted({r["episode_id"] for r in rows if r.get("episode_state") == "COMPLETED"}, key=int)
    replay_dir = output / "replays"
    replay_dir.mkdir(exist_ok=True)
    needed = [eid for eid in eligible if not replay_valid(replay_dir / f"{eid}.json")]
    limit = len(needed) if max_replays is None else max_replays
    selected = needed[:limit]
    state.update(submissions=len(merged), submissions_returned=len(submissions), matches=len(rows),
                 unique_episodes=len({r["episode_id"] for r in rows}), eligible_replays=len(eligible),
                 cached_replays=len(eligible) - len(needed), new_replays=0,
                 deferred_replays=max(0, len(needed) - len(selected)))
    write_json(output / "sync_state.json", state)

    def download(eid: str) -> str:
        fd, name = tempfile.mkstemp(dir=replay_dir, suffix=".partial")
        os.close(fd)
        temporary = Path(name)
        try:
            source.replay(eid, temporary)
            data = validate_replay(temporary)
            reported = data.get("info", {}).get("EpisodeId")
            if reported is not None and str(reported) != eid:
                raise ValueError("Replay の EpisodeId が一致しません")
            os.replace(temporary, replay_dir / f"{eid}.json")
            return eid
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(download, eid): eid for eid in selected}
        for task in as_completed(tasks):
            try:
                task.result()
                state["new_replays"] += 1
            except Exception as exc:
                state["errors"].append({"stage": "replay", "episode_id": tasks[task], "error": safe_error(exc)})
            write_json(output / "sync_state.json", state)
            count = state["new_replays"]
            if count % 10 == 0 or task.exception() is not None:
                progress(f"replays: {count}/{len(selected)}, errors: {len(state['errors'])}")

    state.update(status="partial" if state["errors"] or state["deferred_replays"] else "complete",
                 finished_at=datetime.now(timezone.utc).isoformat(),
                 coverage="API が今回返した履歴と、過去に保存済みの履歴の和集合。未公開・削除済み履歴は保証しない。")
    write_json(output / "sync_state.json", state)
    write_json(snapshot / "sync_state.json", state)
    return state
