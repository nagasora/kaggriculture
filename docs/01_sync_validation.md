# 同期基盤の監査と修正

監査日: 2026-09-10 UTC。

## 既存コードで確認した問題

1. `kaggle competitions replay` の標準出力をReplayファイルとして保存していた。Kaggle 2.2.4のこのコマンドは別ファイルへReplayを保存し、標準出力には保存先の案内文を出す。
2. 提出一覧は1ページしか取得しておらず、既定20件を超える提出が欠落する。
3. `episodes -v` は既定で末尾に案内文を出す。CSVとしての解析に説明文が混ざりうる。
4. CSVの列だけでは `agents[].submissionId/index/reward` を保持できず、相手とseatの確定ができない。
5. ファイルが存在するだけで取得済みとしたため、不正JSONが永続的に残る。
6. 実験IDの `\b(E\d{3})\b` はアンダースコアを含む実際の提出名やE003A/Bを識別できない。
7. mainには前回説明された `src/`、テスト、`.gitignore` が存在しなかった。

## 修正

構造化SDKレスポンス、ページトークン巡回、元APIメタデータ保存、明示的なseat照合、JSON検証後の置換、破損ファイル再取得、履歴の和集合、失敗ID・状態保存、上限付きリトライを実装した。

解析は、最終rewardを使った勝敗と `farms[seat].money` を使ったcoinの推移を分け、両方のrewardが矛盾した場合はUNKNOWNにする。day24・day27のリードから敗北した割合は、各時点のリード件数を分母として集計する。

`EPISODE_TYPE_VALIDATION` は同一提出の自己対戦を含むためmasterに残し、通常成績集計は `EPISODE_TYPE_PUBLIC` かつseatが一意の対戦に限定する。

## 再現可能な検証

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/sync_kaggle_results.py --analyze
python scripts/analyze_kaggle_results.py
```

17テストで、ページング、同一Replay重複、再実行時のダウンロード省略、案内文・ID不一致の修復、中断後の回復、APIから消えた対戦の保持、seat=0/reward=0/欠損の区別、E003A/B識別、逆転時刻、APIとReplayの不一致、validation除外を確認した。

実APIによる初回回収の最終結果は、実行データ内の `sync_state.json` と `analysis/analysis_audit.json` を正とする。認証情報とReplay本文は公開リポジトリに含めない。

## 運用

- 一回同期は上記コマンド。プロセス継続中の繰り返し実行は `--interval-seconds 21600 --analyze`。
- ChatGPTの定期タスクでは、最新の保存アーカイブを作業ディレクトリへ戻し、同じ同期コマンドを実行し、結果アーカイブを更新する。
- Kaggleの認証値は環境変数や権限600の設定ファイルで渡す。URL・HTTPヘッダー・秘密値をエラーログへ書かない。
- APIが返さない過去対戦を存在すると見なさない。`coverage` に取得可能範囲の限界を残す。
- Publicスコアは取得時点の値。同一相手集合で比較していない勝率差から、アルゴリズム改善効果を断定しない。

## 参照

- [Kaggle公式API実装](https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py)
- 実装監査対象: PyPI `kaggle==2.2.4`, `kagglesdk==0.1.37`。
- [元PR #1](https://github.com/nagasora/kaggriculture/pull/1)
