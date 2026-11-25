# Step7 自学习报告（latest）

- 生成时间：2026-11-26 04:38:15
- RunMode：auto_daily
- Today：20261126
- LatestSnapshot：20261125
- LabelUpperBound：20261125

## 1) 最新命中

- trade_date：20261124
- verify_date：20261125
- hit/topn：5/10
- hit_rate：0.5
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：3/5，hit_rate=0.6
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261124.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261111 | 20261112 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20261112 | 20261113 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261113 | 20261116 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20261116 | 20261117 | 1.0 | 0.3333 | 0.2 | 0.3 |
| 20261117 | 20261118 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20261118 | 20261119 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20261119 | 20261120 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20261120 | 20261123 | 1.0 | 1.0 | 0.8 | 0.8 |
| 20261123 | 20261124 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261124 | 20261125 | 1.0 | 0.6667 | 0.6 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 200 | 200 | 98 | 0.49 |
| Top3 | 200 | 600 | 263 | 0.4383 |
| Top5 | 200 | 1000 | 378 | 0.378 |
| Top10 | 200 | 2000 | 637 | 0.3185 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：202
- pass_dates：200
- fail_dates：2
- eligible_train_rows：13327

## 2.1) 样本拒绝分布

- total_rows：13466
- learnable_rows：13327
- rejected_rows：139

| reason | count |
| --- | --- |
| pending_next_snapshot | 139 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：13327
- pos/neg：2429/10898
- feature_coverage：1.0
- pass_trade_dates：200
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
