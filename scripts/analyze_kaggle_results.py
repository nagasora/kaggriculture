#!/usr/bin/env python3
"""How: 保存済みの対戦履歴を再解析する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kaggriculture_sync.analysis import analyze


def main() -> int:
    """How: 対応表を任意指定し、整合性エラーを終了コードに反映する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("05_replays/kaggle_sync"))
    parser.add_argument("--experiment-map", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.experiment_map)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["audit"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
