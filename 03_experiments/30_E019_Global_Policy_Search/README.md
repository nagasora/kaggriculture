# E019 — Global Policy Search Arena

## 0. 目的

E019は「1仮説 = 1実験」を止め、**一つの実験で戦略空間を大域探索する**ためのArenaである。

探索単位は個別のSELL閾値や1つのルールではなく、以下の異なるアルゴリズム族そのものとする。

- 既存強方策 anchors
- phase route mixtures
- opponent-conditioned heuristic routers
- top-replay strategy-family routers
- macro-expert Route-PPO
- E018 learned SELL residual overlay

初期集団は48候補。高速な再構成シミュレータで `48 -> 12 -> 4` に削り、最後のTop4のみをKaggle公式環境で評価してTop3を提出候補にする。

## 1. 48候補

| family | count | 概要 |
|---|---:|---|
| fixed | 12 | E008A/E007/E018D1/E015PPO と E015 expert群 |
| phase_router | 12 | day 0/6/12/18/24 で expertを切替 |
| heuristic_router | 8 | 相手の公開農場情報に反応 |
| replay_router | 8 | 上位Replayのstrategy-family centroidへnearest routing |
| route_ppo | 8 | 720 primitive actionではなく5回のexpert選択をPPO学習 |

## 2. Top replay family discovery

E018B full daily macro corpusを、episode単位でtrain/validation/testを維持したままtrajectory特徴へ集約する。

- K候補: 6 / 8 / 10
- K選択: train silhouetteのみ
- 現在のfull corpusでは K=10 が最良
- router入力は現時点で観測可能な公開状態のみ
- opponent private/future stateは使用しない

## 3. Route-PPO

PPOのaction spaceを720-turnのprimitive actionから8種類の既存macro expertへ変更した。

```text
state -> PPO router -> expert -> 6 days execution -> PPO router -> ...
```

1試合あたり5 decisions。初期global-screen budgetは64–96 episodes/arm。

livestock導入後はcrop-only executorへ戻らないようstate-dependent maskを適用する。

## 4. Arena budgets

- Round 1: 48 candidates × 3 opponents × 2 seeds × both seats = 576 games → keep 12
- Round 2: 12 candidates × 5 opponents × 4 seeds × both seats = 480 games → keep 4
- Official gate: 4 candidates × 4 opponents × 4 fresh seeds × both seats = 128 games → keep 3

`37000–37031` の過去final holdoutは開かない。

## 5. Promotion metric

```text
robust_score =
    0.50 * average win score
  + 0.30 * worst matchup win score
  + 0.15 * median paired score
  + 0.05 * bounded margin tie-break
```

ERROR/DONE率低下は強く減点する。coin marginは主目的にしない。

## 6. 一括実行

Driveの `E019_Global_Search_Arena.ipynb` をColabで上から実行する。

E019では細かいproxy改善に戻らず、Top3に残ったalgorithm familyの周辺だけを次世代populationで掘る。
