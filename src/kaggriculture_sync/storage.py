"""JSON/CSV を一時ファイル経由で保存する。"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_text(path: Path, content: str) -> None:
    """How: 同じディレクトリに書き終えてから置換する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    """How: 非有限数を拒否して UTF-8 JSON を保存する。"""
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path, default: Any = None) -> Any:
    """How: 未作成ファイルだけ既定値にし、破損した索引はエラーにする。"""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """How: 入れ子データを JSON 化し、空でもヘッダーを残す。"""
    import io
    stream = io.StringIO(newline="")
    names = list(dict.fromkeys(fields + [key for row in rows for key in row]))
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    atomic_text(path, stream.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    """How: CSV の引用符内改行を保持して読む。"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_replay(path: Path) -> dict:
    """How: 完了した二人対戦の steps を持つ JSON だけ受理する。"""
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list) or not data["steps"]:
        raise ValueError("Replay に steps がありません")
    reported = data.get("info", {}).get("EpisodeId")
    if path.stem.isdigit() and reported is not None and str(reported) != path.stem:
        raise ValueError("Replay の EpisodeId がファイル名と一致しません")
    terminal = {"DONE", "ERROR", "INVALID", "TIMEOUT"}
    final = data["steps"][-1]
    if not isinstance(final, list) or len(final) != 2 or not all(
        isinstance(agent, dict) and agent.get("status") in terminal for agent in final
    ):
        raise ValueError("二人対戦の完了状態を確認できません")
    return data


def replay_valid(path: Path) -> bool:
    """How: 読めない・途中までの Replay を再取得対象にする。"""
    try:
        validate_replay(path)
        return True
    except (ValueError, OSError, TypeError):
        return False
