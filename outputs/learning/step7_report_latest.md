# Step7 自学习报告（latest）

- 生成时间：2026-09-11 23:11:51
- RunMode：auto_daily
- Today：20260911
- LatestSnapshot：20260911
- LabelUpperBound：20260911

## 1) 最新命中

- trade_date：20260910
- verify_date：20260911
- hit/topn：2/10
- hit_rate：0.2
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260910.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260828 | 20260831 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260831 | 20260901 | 0.0 | 0.3333 | 0.4 | 0.5 |
| 20260901 | 20260902 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260902 | 20260903 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260903 | 20260904 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260904 | 20260907 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260907 | 20260908 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260908 | 20260909 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260909 | 20260910 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260910 | 20260911 | 1.0 | 0.3333 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 153 | 153 | 76 | 0.4967 |
| Top3 | 153 | 459 | 194 | 0.4227 |
| Top5 | 153 | 765 | 286 | 0.3739 |
| Top10 | 153 | 1530 | 482 | 0.315 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：155
- pass_dates：153
- fail_dates：2
- eligible_train_rows：10920

## 2.1) 样本拒绝分布

- total_rows：11052
- learnable_rows：10920
- rejected_rows：132

| reason | count |
| --- | --- |
| pending_next_snapshot | 132 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10920
- pos/neg：1877/9043
- feature_coverage：1.0
- pass_trade_dates：153
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
