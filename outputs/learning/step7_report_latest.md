# Step7 自学习报告（latest）

- 生成时间：2026-09-15 01:04:18
- RunMode：auto_daily
- Today：20260915
- LatestSnapshot：20260914
- LabelUpperBound：20260914

## 1) 最新命中

- trade_date：20260911
- verify_date：20260914
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260911.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260831 | 20260901 | 0.0 | 0.3333 | 0.4 | 0.5 |
| 20260901 | 20260902 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260902 | 20260903 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260903 | 20260904 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260904 | 20260907 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260907 | 20260908 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260908 | 20260909 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260909 | 20260910 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260910 | 20260911 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260911 | 20260914 | 0.0 | 0.3333 | 0.2 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 154 | 154 | 76 | 0.4935 |
| Top3 | 154 | 462 | 195 | 0.4221 |
| Top5 | 154 | 770 | 287 | 0.3727 |
| Top10 | 154 | 1540 | 486 | 0.3156 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：156
- pass_dates：154
- fail_dates：2
- eligible_train_rows：10959

## 2.1) 样本拒绝分布

- total_rows：11100
- learnable_rows：10959
- rejected_rows：141

| reason | count |
| --- | --- |
| pending_next_snapshot | 141 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10959
- pos/neg：1888/9071
- feature_coverage：1.0
- pass_trade_dates：154
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
