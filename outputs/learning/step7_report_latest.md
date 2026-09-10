# Step7 自学习报告（latest）

- 生成时间：2026-09-10 19:24:21
- RunMode：auto_daily
- Today：20260910
- LatestSnapshot：20260910
- LabelUpperBound：20260910

## 1) 最新命中

- trade_date：20260909
- verify_date：20260910
- hit/topn：3/10
- hit_rate：0.3
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：1/5，hit_rate=0.2
- top10：3/10，hit_rate=0.3
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260909.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260827 | 20260828 | 1.0 | 0.3333 | 0.4 | 0.5 |
| 20260828 | 20260831 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260831 | 20260901 | 0.0 | 0.3333 | 0.4 | 0.5 |
| 20260901 | 20260902 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260902 | 20260903 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260903 | 20260904 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260904 | 20260907 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260907 | 20260908 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260908 | 20260909 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260909 | 20260910 | 0.0 | 0.0 | 0.2 | 0.3 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 152 | 152 | 75 | 0.4934 |
| Top3 | 152 | 456 | 193 | 0.4232 |
| Top5 | 152 | 760 | 285 | 0.375 |
| Top10 | 152 | 1520 | 480 | 0.3158 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：154
- pass_dates：152
- fail_dates：2
- eligible_train_rows：10886

## 2.1) 样本拒绝分布

- total_rows：11013
- learnable_rows：10886
- rejected_rows：127

| reason | count |
| --- | --- |
| pending_next_snapshot | 127 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10886
- pos/neg：1870/9016
- feature_coverage：1.0
- pass_trade_dates：152
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
