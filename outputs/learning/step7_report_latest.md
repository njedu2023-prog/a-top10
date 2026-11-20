# Step7 自学习报告（latest）

- 生成时间：2026-11-21 03:06:40
- RunMode：auto_daily
- Today：20261121
- LatestSnapshot：20261120
- LabelUpperBound：20261120

## 1) 最新命中

- trade_date：20261119
- verify_date：20261120
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：0/5，hit_rate=0.0
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261119.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261106 | 20261109 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20261109 | 20261110 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261110 | 20261111 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20261111 | 20261112 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20261112 | 20261113 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261113 | 20261116 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20261116 | 20261117 | 1.0 | 0.3333 | 0.2 | 0.3 |
| 20261117 | 20261118 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20261118 | 20261119 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20261119 | 20261120 | 0.0 | 0.0 | 0.0 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 197 | 197 | 96 | 0.4873 |
| Top3 | 197 | 591 | 257 | 0.4349 |
| Top5 | 197 | 985 | 369 | 0.3746 |
| Top10 | 197 | 1970 | 621 | 0.3152 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：199
- pass_dates：197
- fail_dates：2
- eligible_train_rows：13138

## 2.1) 样本拒绝分布

- total_rows：13256
- learnable_rows：13138
- rejected_rows：118

| reason | count |
| --- | --- |
| pending_next_snapshot | 118 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：13138
- pos/neg：2375/10763
- feature_coverage：1.0
- pass_trade_dates：197
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
