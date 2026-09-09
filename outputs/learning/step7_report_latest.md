# Step7 自学习报告（latest）

- 生成时间：2026-09-09 23:16:53
- RunMode：auto_daily
- Today：20260909
- LatestSnapshot：20260909
- LabelUpperBound：20260909

## 1) 最新命中

- trade_date：20260908
- verify_date：20260909
- hit/topn：2/10
- hit_rate：0.2
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260908.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260826 | 20260827 | 1.0 | 0.6667 | 0.8 | 0.4 |
| 20260827 | 20260828 | 1.0 | 0.3333 | 0.4 | 0.5 |
| 20260828 | 20260831 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260831 | 20260901 | 0.0 | 0.3333 | 0.4 | 0.5 |
| 20260901 | 20260902 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260902 | 20260903 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260903 | 20260904 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260904 | 20260907 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260907 | 20260908 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260908 | 20260909 | 0.0 | 0.3333 | 0.4 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 151 | 151 | 75 | 0.4967 |
| Top3 | 151 | 453 | 193 | 0.426 |
| Top5 | 151 | 755 | 284 | 0.3762 |
| Top10 | 151 | 1510 | 477 | 0.3159 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：153
- pass_dates：151
- fail_dates：2
- eligible_train_rows：10838

## 2.1) 样本拒绝分布

- total_rows：10979
- learnable_rows：10838
- rejected_rows：141

| reason | count |
| --- | --- |
| pending_next_snapshot | 141 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10838
- pos/neg：1861/8977
- feature_coverage：1.0
- pass_trade_dates：151
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
