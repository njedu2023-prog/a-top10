# Step7 自学习报告（latest）

- 生成时间：2026-09-03 19:21:00
- RunMode：auto_daily
- Today：20260903
- LatestSnapshot：20260903
- LabelUpperBound：20260903

## 1) 最新命中

- trade_date：20260902
- verify_date：20260903
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260902.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260820 | 20260821 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260821 | 20260824 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260824 | 20260825 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260825 | 20260826 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260826 | 20260827 | 1.0 | 0.6667 | 0.8 | 0.4 |
| 20260827 | 20260828 | 1.0 | 0.3333 | 0.4 | 0.5 |
| 20260828 | 20260831 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260831 | 20260901 | 0.0 | 0.3333 | 0.4 | 0.5 |
| 20260901 | 20260902 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260902 | 20260903 | 0.0 | 0.3333 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 147 | 147 | 74 | 0.5034 |
| Top3 | 147 | 441 | 188 | 0.4263 |
| Top5 | 147 | 735 | 278 | 0.3782 |
| Top10 | 147 | 1470 | 466 | 0.317 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：149
- pass_dates：147
- fail_dates：2
- eligible_train_rows：10600

## 2.1) 样本拒绝分布

- total_rows：10735
- learnable_rows：10600
- rejected_rows：135

| reason | count |
| --- | --- |
| pending_next_snapshot | 135 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10600
- pos/neg：1807/8793
- feature_coverage：1.0
- pass_trade_dates：147
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
