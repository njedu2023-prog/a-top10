# Step7 自学习报告（latest）

- 生成时间：2026-10-13 19:25:00
- RunMode：auto_daily
- Today：20261013
- LatestSnapshot：20261013
- LabelUpperBound：20261013

## 1) 最新命中

- trade_date：20261012
- verify_date：20261013
- hit/topn：4/10
- hit_rate：0.4
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261012.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260921 | 20260922 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260922 | 20260923 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260923 | 20260924 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260924 | 20260928 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260928 | 20260929 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260929 | 20260930 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260930 | 20261008 | 1.0 | 0.6667 | 0.6 | 0.6 |
| 20261008 | 20261009 | 0.0 | 0.6667 | 0.4 | 0.4 |
| 20261009 | 20261012 | 1.0 | 0.6667 | 0.6 | 0.3 |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 169 | 169 | 84 | 0.497 |
| Top3 | 169 | 507 | 219 | 0.432 |
| Top5 | 169 | 845 | 318 | 0.3763 |
| Top10 | 169 | 1690 | 534 | 0.316 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：171
- pass_dates：169
- fail_dates：2
- eligible_train_rows：11748

## 2.1) 样本拒绝分布

- total_rows：11904
- learnable_rows：11748
- rejected_rows：156

| reason | count |
| --- | --- |
| pending_next_snapshot | 156 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11748
- pos/neg：2071/9677
- feature_coverage：1.0
- pass_trade_dates：169
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
