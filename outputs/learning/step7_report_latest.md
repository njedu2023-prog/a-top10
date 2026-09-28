# Step7 自学习报告（latest）

- 生成时间：2026-09-29 02:44:46
- RunMode：auto_daily
- Today：20260929
- LatestSnapshot：20260928
- LabelUpperBound：20260928

## 1) 最新命中

- trade_date：20260924
- verify_date：20260928
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260924.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260911 | 20260914 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260914 | 20260915 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20260915 | 20260916 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260916 | 20260917 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260917 | 20260918 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20260918 | 20260921 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260921 | 20260922 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260922 | 20260923 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260923 | 20260924 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260924 | 20260928 | 0.0 | 0.3333 | 0.2 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 163 | 163 | 79 | 0.4847 |
| Top3 | 163 | 489 | 208 | 0.4254 |
| Top5 | 163 | 815 | 305 | 0.3742 |
| Top10 | 163 | 1630 | 513 | 0.3147 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：165
- pass_dates：163
- fail_dates：2
- eligible_train_rows：11479

## 2.1) 样本拒绝分布

- total_rows：11603
- learnable_rows：11479
- rejected_rows：124

| reason | count |
| --- | --- |
| pending_next_snapshot | 124 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11479
- pos/neg：2002/9477
- feature_coverage：1.0
- pass_trade_dates：163
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
