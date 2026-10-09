# Step7 自学习报告（latest）

- 生成时间：2026-10-09 21:58:16
- RunMode：auto_daily
- Today：20261009
- LatestSnapshot：20261009
- LabelUpperBound：20261009

## 1) 最新命中

- trade_date：20261008
- verify_date：20261009
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：2/3，hit_rate=0.6667
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261008.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260917 | 20260918 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20260918 | 20260921 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260921 | 20260922 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260922 | 20260923 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260923 | 20260924 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260924 | 20260928 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260928 | 20260929 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260929 | 20260930 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260930 | 20261008 | 1.0 | 0.6667 | 0.6 | 0.6 |
| 20261008 | 20261009 | 0.0 | 0.6667 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 167 | 167 | 82 | 0.491 |
| Top3 | 167 | 501 | 215 | 0.4291 |
| Top5 | 167 | 835 | 313 | 0.3749 |
| Top10 | 167 | 1670 | 527 | 0.3156 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：169
- pass_dates：167
- fail_dates：2
- eligible_train_rows：11647

## 2.1) 样本拒绝分布

- total_rows：11802
- learnable_rows：11647
- rejected_rows：155

| reason | count |
| --- | --- |
| pending_next_snapshot | 155 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11647
- pos/neg：2043/9604
- feature_coverage：1.0
- pass_trade_dates：167
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
