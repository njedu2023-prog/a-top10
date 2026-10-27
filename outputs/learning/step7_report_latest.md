# Step7 自学习报告（latest）

- 生成时间：2026-10-28 00:52:11
- RunMode：auto_daily
- Today：20261028
- LatestSnapshot：20261027
- LabelUpperBound：20261027

## 1) 最新命中

- trade_date：20261026
- verify_date：20261027
- hit/topn：5/10
- hit_rate：0.5
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：3/5，hit_rate=0.6
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261026.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |
| 20261020 | 20261021 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261021 | 20261022 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20261022 | 20261023 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261023 | 20261026 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20261026 | 20261027 | 0.0 | 0.3333 | 0.6 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 179 | 179 | 88 | 0.4916 |
| Top3 | 179 | 537 | 235 | 0.4376 |
| Top5 | 179 | 895 | 339 | 0.3788 |
| Top10 | 179 | 1790 | 573 | 0.3201 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：181
- pass_dates：179
- fail_dates：2
- eligible_train_rows：12205

## 2.1) 样本拒绝分布

- total_rows：12336
- learnable_rows：12205
- rejected_rows：131

| reason | count |
| --- | --- |
| pending_next_snapshot | 131 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12205
- pos/neg：2182/10023
- feature_coverage：1.0
- pass_trade_dates：179
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
