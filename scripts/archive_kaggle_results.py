#!/usr/bin/env python3
"""How: 同期データのZIP保存・復元をコマンドで実行する。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kaggriculture_sync.archive import pack, restore


def main() -> None:
    """How: 操作、アーカイブ、作業ディレクトリを明示指定する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['pack', 'restore'])
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('05_replays/kaggle_sync'))
    args = parser.parse_args()
    result = pack(args.root, args.archive) if args.operation == 'pack' else restore(args.archive, args.root)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
