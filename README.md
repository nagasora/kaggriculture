# kaggriculture

Kaggle の提出・対戦・Replay を取得し、実験別の成績と coin 時系列を更新する。

```bash
python -m pip install -e '.[test]'
python scripts/sync_kaggle_results.py --analyze
```

認証は実行環境の `~/.kaggle/kaggle.json`、`KAGGLE_CONFIG_DIR`、または `KAGGLE_API_TOKEN` を使用する。認証ファイル・秘密値をリポジトリに追加しない。

- [同期と解析の仕様](05_replays/kaggle_sync/README.md)
- [検証・運用ノート](docs/01_sync_validation.md)

Colab などの継続実行環境では、永続ディレクトリを指定して繰り返し同期できる。

```bash
python scripts/sync_kaggle_results.py \
  --output-dir /content/drive/MyDrive/kaggriculture_sync \
  --interval-seconds 21600 --analyze
```

このコマンドはプロセスが動いている間だけ継続する。Colab の切断後まで動くスケジューラーではない。ChatGPT の定期タスクを使う場合は、保存済みアーカイブを復元してから同じ一回同期コマンドを実行する。
