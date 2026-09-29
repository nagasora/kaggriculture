# E045 — Hierarchical Expert-Transformer RL

## 方針

E043の719手coherent planを捨てない。E042で失敗した24手ごとのfull-route切替を禁止し、3日=72手の自然境界かつday6までだけfull-route候補を評価する。day6以降はE043のfield/handsを固定し、市場順序・front-run・流動性保護だけを残差actionにする。

Transformerの役割は「どのfamilyかを分類すること」ではなく、公開状態履歴と候補expertを条件にした Q(history, option) の推定である。family分類probeでは現在状態MLPがTransformerを上回ったため、sequence modelを価値予測へ移した。

## 既存実験からの制約

- E043: part043完全holdout 96試合で84勝12敗、87.5%。長いcoherent trajectory + CompiledPlan lookahead=4が現在のbase。
- E042: 24手switchは工程因果を破壊して失敗。SELL residualは勝敗を改善せず、residual PPOはKEEPへcollapse。
- E025: Temporal-16 supervised accuracyは強かったが、fixed stride / correction非DAgger / PPO不足 / cache契約に問題。
- 最新40 top trajectories: same-family exact matchはblock0 63.6%, block1 27.1%, block2 7.3%。一方3日macro署名でfamilyはblock0 90.0%, block1 87.5%。
- E045 representation probe: E037 seed-disjoint testでcurrent MLP 94.69%, causal Transformer 92.19%。family identificationにTransformerを使う理由はない。

## Architecture

1. Base executor: E043 Majkel1337 trajectory + CompiledPlan repair。
2. Expert bank: E043 + latest top replay routes + current high-performing public route families。
3. Macro selector: 72-turn boundaries, full-route switching only through step144. HierarchicalOptionTransformer predicts candidate-conditioned Q。
4. Compatibility mask: worker / structure / resource prerequisites that cannot be repaired cheaply are hard mask。low confidenceはbase E043へfallback。
5. Residual market controller: day6以降、field/handsは変更せず KEEP / ORDER_ONLY / FRONT_RUN / LIQUIDITY_GUARD だけ選ぶ。
6. Reactive guard layer: hand alignment, weed repair, sell clamp, terminal liquidationを常時保持。budget/room/dead-stock/front-runはablationして採用する。

## Learning

### Stage 1 — imitation / representation

最新top replayをepisode単位でsplitし、future observationを使わずstate-historyとexpert action/styleを学習する。固定strideではなくday/shop boundary・投資・売買・資源不足イベントを必ず残す。

### Stage 2 — true DAgger + counterfactual AWR

learner自身をleagueへrolloutし、実際に到達したmacro境界を保存する。同一state snapshotから互換optionをforkして72手評価し、その後base continuationでterminal relative cashまで測る。これを教師returnとしてAWR/value regressionする。teacher trajectoryを追加するだけのE025型correctionはDAggerと呼ばない。

### Stage 3 — league PPO

Stage 2がE043をfresh leagueで上回った場合だけ実施。PPO actionはexpert/residual optionだけ。低レベル719-step actionを直接学習しない。KL to AWR policyを強く掛け、KEEP/base optionを明示する。

## Opponent league

- current public high-score agents: adaptive market-aware / Multi-Route / Ahmed V43 / reactive shop-router / rank-your-agent lineage。
- latest synchronized top replay proxies。
- replay fragmentsから合成したreactive opponents。
- self-play snapshots。

G4/E008Aは一要素であり主評価基準にしない。

## Promotion gate

1. alignment / no-future / split / loader / deterministic retry。
2. unknown episode/seedでrepresentationが崩れない。
3. fresh seed × both seats × heterogeneous league。
4. E043より勝敗点 +3pt以上。
5. worst-familyはE043から非退行。
6. market/moonの既知弱点を個別に改善。
7. Kaggle提出は自動で行わない。候補artifactを作るだけ。

## Current evidence

最新40 trajectoryの解析ではsame-family longest common prefixは平均30.6手、cross-familyは3.19手。72手blockのexact一致は同familyでも63.6% -> 27.1% -> 7.3%と急減する。このためraw tapeの途中接合ではなく、macro option + observation-dependent executorを採用する。

E037因果整列済み特徴でfamily-classification probeを行うと、current-state MLP=94.69%、causal Transformer=92.19%。よってTransformerはfamily classifierではなくoption-value / opponent-dynamics modelとして使う。

ローカル安全テストは8件すべて成功。公式Kaggle提出は0件。

## Run

    PYTHONPATH=src python scripts/e045_analyze_plan_bank.py --bank /path/to/top5_plan_bank.json --out runs/e045_macro_bank_analysis.json
    pytest -q tests/test_e045_hierarchical.py

次の主要runはcounterfactual dataset生成 → AWR/value学習 → fresh league評価であり、PPOはその後。
