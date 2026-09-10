# Kaggle result sync

提出履歴、各 submission の episode、replay を差分同期するための領域です。

## 実行

Kaggle CLI の認証を設定した環境で次を実行します。

```bash
python scripts/sync_kaggle_results.py
```

生成物は `submissions.csv`, `episodes/`, `replays/`, `match_index.csv`, `sync_state.json` です。

Replay は episode ID 単位で保存し、既存ファイルは再取得しません。したがって初回は過去対戦を回収し、2回目以降は新しい対戦だけを追加取得できます。

認証ファイルはソース管理へ含めないでください。
