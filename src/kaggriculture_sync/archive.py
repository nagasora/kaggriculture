"""取得データを秘密情報のないZIPにまとめ、差分同期用に復元する。"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {'submissions.json', 'submissions.csv', 'match_index.csv', 'sync_state.json'}
ANALYSIS_FILES = {'match_master.csv', 'coin_timeseries.csv', 'daily_coin_margin.csv',
                  'experiment_summary.csv', 'experiment_summary.json', 'submission_summary.csv',
                  'seat_summary.csv', 'opponent_summary.csv', 'failure_index.csv', 'analysis_audit.json'}


def allowed(name: str) -> bool:
    """How: 同期仕様で定義したデータパスだけを許可する。"""
    if '\\' in name or '..' in PurePosixPath(name).parts or PurePosixPath(name).is_absolute():
        return False
    parts = PurePosixPath(name).parts
    if len(parts) == 1:
        return name in ROOT_FILES
    if len(parts) == 2:
        folder, file = parts
        return ((folder == 'analysis' and file in ANALYSIS_FILES) or
                (folder == 'replays' and bool(re.fullmatch(r'\d+\.json', file))) or
                (folder == 'episodes' and bool(re.fullmatch(r'\d+\.(json|csv)', file))))
    if len(parts) == 3 and parts[0] == 'snapshots':
        return bool(re.fullmatch(r'[\dT+Z.:-]+', parts[1])) and (
            parts[2] in {'submissions.json', 'sync_state.json'} or
            bool(re.fullmatch(r'episodes_\d+\.json', parts[2])))
    return False


def pack(root: Path, destination: Path) -> dict:
    """How: 許可リストに一致する通常ファイルを圧縮し、完了後に置換する。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix='.partial')
    os.close(fd)
    count = 0
    try:
        with zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(root.rglob('*')):
                relative = path.relative_to(root).as_posix()
                if path.is_file() and not path.is_symlink() and allowed(relative):
                    archive.write(path, relative)
                    count += 1
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return {'files': count, 'bytes': destination.stat().st_size}


def restore(archive_path: Path, root: Path) -> int:
    """How: 全メンバーを検査し、許可パスだけを一時ファイルから復元する。"""
    root.mkdir(parents=True, exist_ok=True)
    base = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for item in members:
            target = base / item.filename
            if not allowed(item.filename) or not target.resolve().is_relative_to(base):
                raise ValueError('同期アーカイブに許可されないパスがあります')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('シンボリックリンクは復元しません')
        for item in members:
            target = base / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=target.parent, suffix='.partial')
            try:
                with os.fdopen(fd, 'wb') as destination, archive.open(item) as source:
                    shutil.copyfileobj(source, destination)
                os.replace(name, target)
            finally:
                Path(name).unlink(missing_ok=True)
    return len(members)
