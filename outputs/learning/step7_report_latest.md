# Step7 自学习报告（latest）

- 生成时间：2026-10-19 19:22:47
- RunMode：auto_daily
- Today：20261019
- LatestSnapshot：20261019
- LabelUpperBound：20261019

## 1) 最新命中

- trade_date：20261016
- verify_date：20261019
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261016.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260928 | 20260929 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260929 | 20260930 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260930 | 20261008 | 1.0 | 0.6667 | 0.6 | 0.6 |
| 20261008 | 20261009 | 0.0 | 0.6667 | 0.4 | 0.4 |
| 20261009 | 20261012 | 1.0 | 0.6667 | 0.6 | 0.3 |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 173 | 173 | 85 | 0.4913 |
| Top3 | 173 | 519 | 224 | 0.4316 |
| Top5 | 173 | 865 | 324 | 0.3746 |
| Top10 | 173 | 1730 | 546 | 0.3156 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：175
- pass_dates：173
- fail_dates：2
- eligible_train_rows：11934

## 2.1) 样本拒绝分布

- total_rows：12067
- learnable_rows：11934
- rejected_rows：133

| reason | count |
| --- | --- |
| pending_next_snapshot | 133 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11934
- pos/neg：2107/9827
- feature_coverage：1.0
- pass_trade_dates：173
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
