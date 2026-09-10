#!/usr/bin/env python3
"""How: 既存コマンドから差分同期・任意の繰り返し実行を起動する。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaggriculture_sync.client import KaggleSource
from kaggriculture_sync.sync import safe_error, sync


def main() -> int:
    """How: 一回同期か指定間隔の同期を選び、失敗は終了コードに反映する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("05_replays/kaggle_sync"))
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--max-replays", type=int)
    parser.add_argument("--interval-seconds", type=int, default=0)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.max_replays is not None and args.max_replays < 0:
        parser.error("--max-replays は0以上にしてください")
    if args.interval_seconds and args.interval_seconds < 300:
        parser.error("同期間隔は300秒以上にしてください")
    while True:
        try:
            state = sync(KaggleSource(), args.output_dir, args.competition, args.workers, args.max_replays)
            print(json.dumps(state, ensure_ascii=False))
            code = 0 if state["status"] == "complete" else 2
            if args.analyze:
                from kaggriculture_sync.analysis import analyze
                print(json.dumps(analyze(args.output_dir), ensure_ascii=False))
        except Exception as exc:
            print(f"同期失敗: {safe_error(exc)}", file=sys.stderr)
            code = 1
        if not args.interval_seconds:
            return code
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
