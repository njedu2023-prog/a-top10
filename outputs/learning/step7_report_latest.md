# Step7 自学习报告（latest）

- 生成时间：2026-07-29 19:16:32
- RunMode：auto_daily
- Today：20260729
- LatestSnapshot：20260729
- LabelUpperBound：20260729

## 1) 最新命中

- trade_date：20260728
- verify_date：20260729
- hit/topn：2/10
- hit_rate：0.2
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260728.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260715 | 20260716 | 0.0 | 0.6667 | 0.6 | 0.4 |
| 20260716 | 20260717 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260717 | 20260720 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260720 | 20260721 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260721 | 20260722 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260722 | 20260723 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20260723 | 20260724 | 1.0 | 0.3333 | 0.4 | 0.4 |
| 20260724 | 20260727 | 1.0 | 0.6667 | 0.8 | 0.5 |
| 20260727 | 20260728 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260728 | 20260729 | 0.0 | 0.0 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 121 | 121 | 64 | 0.5289 |
| Top3 | 121 | 363 | 163 | 0.449 |
| Top5 | 121 | 605 | 241 | 0.3983 |
| Top10 | 121 | 1210 | 395 | 0.3264 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：123
- pass_dates：121
- fail_dates：2
- eligible_train_rows：8813

## 2.1) 样本拒绝分布

- total_rows：8987
- learnable_rows：8813
- rejected_rows：174

| reason | count |
| --- | --- |
| pending_next_snapshot | 174 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8813
- pos/neg：1446/7367
- feature_coverage：1.0
- pass_trade_dates：121
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
