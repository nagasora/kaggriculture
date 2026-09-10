#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def run_kaggle(*args: str) -> str:
    # Why not shell=True: IDs and arguments must never be interpreted by a shell.
    result = subprocess.run(["kaggle", *args], check=True, capture_output=True, text=True)
    return result.stdout


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def get_id(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    normalized = {k.lower().replace("_", "").replace(" ", ""): v for k, v in row.items()}
    for candidate in candidates:
        value = normalized.get(candidate.lower().replace("_", "").replace(" ", ""))
        if value:
            return value.strip()
    return None


def sync(output: Path, competition: str = "kaggriculture") -> dict[str, int]:
    # How: refresh indexes every run and fetch only replay files that are still missing.
    episode_dir = output / "episodes"
    replay_dir = output / "replays"
    episode_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    submission_text = run_kaggle("competitions", "submissions", competition, "-v")
    (output / "submissions.csv").write_text(submission_text, encoding="utf-8")
    submissions = rows(submission_text)
    matches: list[dict[str, str]] = []
    downloaded = 0

    for submission in submissions:
        submission_id = get_id(submission, ("submissionId", "id", "ref"))
        if not submission_id:
            continue
        episode_text = run_kaggle("competitions", "episodes", submission_id, "-v")
        (episode_dir / f"{submission_id}.csv").write_text(episode_text, encoding="utf-8")
        for episode in rows(episode_text):
            episode_id = get_id(episode, ("episodeId", "id"))
            if not episode_id:
                continue
            matches.append({"submission_id": submission_id, **episode})
            replay_path = replay_dir / f"{episode_id}.json"
            if not replay_path.exists():
                replay_path.write_text(run_kaggle("competitions", "replay", episode_id), encoding="utf-8")
                downloaded += 1

    index = output / "match_index.csv"
    if matches:
        fields = list(dict.fromkeys(k for row in matches for k in row))
        with index.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(matches)
    else:
        index.write_text("", encoding="utf-8")

    state = {"submissions": len(submissions), "episodes": len(matches), "new_replays": downloaded}
    (output / "sync_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("05_replays/kaggle_sync"))
    args = parser.parse_args()
    state = sync(args.output_dir)
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
