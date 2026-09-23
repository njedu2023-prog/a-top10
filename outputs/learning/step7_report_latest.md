# Step7 自学习报告（latest）

- 生成时间：2026-09-23 23:35:10
- RunMode：auto_daily
- Today：20260923
- LatestSnapshot：20260923
- LabelUpperBound：20260923

## 1) 最新命中

- trade_date：20260922
- verify_date：20260923
- hit/topn：5/10
- hit_rate：0.5
- top1：0/1，hit_rate=0.0
- top3：2/3，hit_rate=0.6667
- top5：3/5，hit_rate=0.6
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260922.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260909 | 20260910 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260910 | 20260911 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260911 | 20260914 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260914 | 20260915 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20260915 | 20260916 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260916 | 20260917 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260917 | 20260918 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20260918 | 20260921 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260921 | 20260922 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260922 | 20260923 | 0.0 | 0.6667 | 0.6 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 161 | 161 | 78 | 0.4845 |
| Top3 | 161 | 483 | 205 | 0.4244 |
| Top5 | 161 | 805 | 301 | 0.3739 |
| Top10 | 161 | 1610 | 508 | 0.3155 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：163
- pass_dates：161
- fail_dates：2
- eligible_train_rows：11388

## 2.1) 样本拒绝分布

- total_rows：11525
- learnable_rows：11388
- rejected_rows：137

| reason | count |
| --- | --- |
| pending_next_snapshot | 137 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11388
- pos/neg：1982/9406
- feature_coverage：1.0
- pass_trade_dates：161
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
