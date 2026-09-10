# 提出・対戦の同期

## 実行

```bash
python -m pip install -e '.[test]'
python scripts/sync_kaggle_results.py --analyze
python scripts/analyze_kaggle_results.py
python -m pytest -q
```

`--output-dir` / `--root` で保存場所を変更する。`--workers` は1〜4、既定3。`--max-replays 5` は取得件数を限定した監査用で、残件があれば `partial` として終了コード2を返す。`--interval-seconds 21600 --analyze` はプロセスの稼働中に6時間ごとの同期を行う。

## 保存物

| パス | 内容 |
|---|---|
| `submissions.json` / `.csv` | 提出一覧。ページトークンを最後まで辿る |
| `episodes/<submission_id>.json` / `.csv` | agent・seat・rewardを含む対戦メタデータ |
| `replays/<episode_id>.json` | 公式APIから取得した、検証済みのReplay本文 |
| `match_index.csv` | `(submission_id, episode_id)` を一意キーとする索引 |
| `sync_state.json` | 完了/一部失敗・取得件数・失敗ID |
| `snapshots/<UTC日時>/` | 取得時点の提出スコア・APIが返した対戦範囲 |
| `analysis/match_master.csv` | 勝敗・相手・seat・coin差・逆転時刻・店舗・残在庫 |
| `analysis/coin_timeseries.csv` | 全stepのcoin時系列 |
| `analysis/daily_coin_margin.csv` | 各dayの最後に観測したcoin差 |
| `analysis/experiment_summary.csv` / `.json` | E001/E003A/E003B等の別々の集計 |
| `analysis/submission_summary.csv` | 提出ID別成績 |
| `analysis/seat_summary.csv` / `opponent_summary.csv` | 実験×seat、実験×相手別成績 |
| `analysis/failure_index.csv` | 負け・引き分けだけの参照用索引 |
| `analysis/analysis_audit.json` | UNKNOWN件数、Replay整合性エラー |

## 判定の定義

- `WIN/LOSS/DRAW` は完了した対戦の、自分と相手の **reward** を比較する。欠損を0に置き換えない。
- 自分のseatは `agents[].submissionId` と `agents[].index` で照合する。配列の順序に依存しない。SDKの既定値省略では `index=0` が消えるため、保存時に既定値を保持する。
- `my_final_coins` / `opponent_final_coins` は **最終フレーム**の `observation.farms[seat].money`。rewardと別の列に保存する。
- APIとReplayのrewardが矛盾すれば `UNKNOWN/reward_mismatch` とし、監査ファイルに残す。
- dayはゲーム観測の **0始まり**。`coin_margin_day24/27` はそれぞれ `day=24/27, hour=0` の観測値。該当観測がなければ欠損。
- `reversal_losses_day24/27` はその時点でcoin差が正、かつ最終結果がLOSSだった件数。
- `reversal_rate_among_leads_day24/27` の分母は、その時点でリードし、勝敗も検証できた対戦数。全対戦数や全敗北数ではない。
- `decisive_reversal_step/day` は最終結果LOSSの対戦で、最後の正のcoin差の後、最初に負になった観測。因果関係を示す指標ではない。
- 通常成績の集計対象は `EPISODE_TYPE_PUBLIC` かつ自分のseatが一意の対戦。validation・自己対戦等はmasterに保持し、通常勝率から除く。
- `win_rate` は `WIN/(WIN+LOSS+DRAW)`。UNKNOWNを分母から除外する。相手の強さ・時期が異なるため、実験間の因果的な優劣とは解釈しない。
- `failure_index.csv` は分析用の敗北コーパス。ここから全体勝率を推定しない。

## 実験IDの対応

`kaggriculture_e001_submission.tar.gz` → E001、`02_E003A_submission.tar.gz` → E003A、`03_E003B_primary_submission.tar.gz` → E003B。

ファイル名と説明から一意に判別できない場合はUNMAPPED。任意のJSON `{ "submission_id": "E004" }` を `--experiment-map mapping.json` に渡せば明示指定できる。提出が存在しない実験の成績は生成しない。

## 再実行と取得範囲

ReplayはJSONの構造・完了状態・EpisodeIdを検証してからファイルを確定する。既存Replayが空、CLI案内文、途中のJSON、ID不一致なら再取得する。別々の提出に同じ対戦が現れてもReplayは1回だけ保存する。

提出一覧は全ページ取得する。対戦API `ListSubmissionEpisodes` は現行SDKにページ引数がない。**APIが返す全件を回収するが、削除済み・非公開・APIが返さない過去対戦まで全件あるとは保証しない。** 以前取得した対戦は次回APIから消えても保持する。

通信障害・429・5xxは上限付きで再試行し、401/403などは無限に再試行しない。失敗は終了コードと状態に出し、次回は残件を再取得する。初回は数GBになりうる。公開GitHubにはデータをcommitせず、Drive等の永続保存先で運用する。

## アーカイブの保存と復元

```bash
python scripts/archive_kaggle_results.py pack --archive /path/kaggriculture_sync_latest.zip
python scripts/archive_kaggle_results.py restore --archive /path/kaggriculture_sync_latest.zip
```

アーカイブは同期データの許可パスだけを含む。認証ファイルは対象外。復元は既存索引・Replayを更新するため、同期プロセス停止中に実行する。
