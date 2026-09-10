#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT = re.compile(r"\b(E\d{3})\b", re.I)


def load(path):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm(row):
    return {k.lower().replace("_", "").replace(" ", ""): v for k, v in row.items()}


def get(row, *names):
    data = norm(row)
    for name in names:
        result = data.get(name.lower().replace("_", "").replace(" ", ""))
        if result:
            return result
    return ""


def exp_id(row):
    # How: map E001/E002/E003 from submission metadata rather than hard-coded IDs.
    match = EXPERIMENT.search(" ".join(str(v or "") for v in row.values()))
    return match.group(1).upper() if match else "UNMAPPED"


def outcome(row):
    # Why not infer unknown Kaggle fields: unknown semantics stay explicitly UNKNOWN.
    result = get(row, "result", "outcome", "status").upper()
    if result in {"WIN", "WON", "VICTORY"}:
        return "WIN"
    if result in {"LOSS", "LOSE", "LOST", "DEFEAT"}:
        return "LOSS"
    if result in {"DRAW", "TIE"}:
        return "DRAW"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("05_replays/kaggle_sync"))
    args = parser.parse_args()
    submissions = load(args.root / "submissions.csv")
    mapping = {get(r, "submissionId", "id", "ref"): exp_id(r) for r in submissions}
    matches = load(args.root / "match_index.csv")
    output = []
    counts = defaultdict(Counter)
    for row in matches:
        sid = get(row, "submission_id", "submissionId")
        eid = get(row, "episodeId", "id")
        exp = mapping.get(sid, "UNMAPPED")
        result = outcome(row)
        counts[exp][result] += 1
        output.append({"experiment_id": exp, "submission_id": sid, "episode_id": eid, "outcome": result, **row})
    analysis = args.root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    if output:
        fields = list(dict.fromkeys(k for row in output for k in row))
        with (analysis / "match_master.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)
    summary = {exp: {"episodes": sum(c.values()), **dict(c)} for exp, c in sorted(counts.items())}
    (analysis / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
