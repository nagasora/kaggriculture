# E019 Global Policy Search Arena — 実装状況

作成日: 2026-09-17 JST

## 結論

E019はE018D1を順番に改善する直列実験ではなく、**48個の異なる方策を一度に比較する大域探索基盤**へ切り替えた。

実装済み:

- 48候補manifest
- Top replay trajectory clustering
- replay-family / opponent-conditioned / phase routers
- E018 SELL residual overlay
- 5-decision Route-PPO
- reconstructed simulator parallel arena
- robust promotion score
- Top4のみのKaggle公式環境gate
- standalone submission tar packaging
- Colab one-command bundle（Drive側）

## Top replay family discovery

E018B full daily macro corpus 47,160 daily rows / 1,335 winning episode-seatsを集約した。

train silhouetteは K=6: 0.11609、K=8: 0.13082、K=10: **0.13547** で、K=10を採用した。

## Smoke validation

- 代表5 family × both seatsの並列runner: 10/10 DONE
- Route-PPO rollout/update: 完走
- livestock導入後のcrop-only expertへの不正switchを検出し、state-dependent action maskで修正
- replay-router standalone submission tar: 719 calls完走
- unit tests: 5 passed

## Full Arena

- Round1: 48 -> 12, 576 reconstructed games
- Round2: 12 -> 4, 480 reconstructed games
- Final: 4 -> 3, 128 official-environment games

この実行環境には外部pip accessがないため公式environment gateのfull数値は未生成。DriveのColab notebookで一括実行する。
