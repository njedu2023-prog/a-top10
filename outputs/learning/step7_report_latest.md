# Step7 自学习报告（latest）

- 生成时间：2026-09-16 19:20:36
- RunMode：auto_daily
- Today：20260916
- LatestSnapshot：20260916
- LabelUpperBound：20260916

## 1) 最新命中

- trade_date：20260915
- verify_date：20260916
- hit/topn：5/10
- hit_rate：0.5
- top1：0/1，hit_rate=0.0
- top3：2/3，hit_rate=0.6667
- top5：3/5，hit_rate=0.6
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260915.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260902 | 20260903 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260903 | 20260904 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260904 | 20260907 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260907 | 20260908 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260908 | 20260909 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260909 | 20260910 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260910 | 20260911 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260911 | 20260914 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260914 | 20260915 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20260915 | 20260916 | 0.0 | 0.6667 | 0.6 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 156 | 156 | 76 | 0.4872 |
| Top3 | 156 | 468 | 199 | 0.4252 |
| Top5 | 156 | 780 | 292 | 0.3744 |
| Top10 | 156 | 1560 | 493 | 0.316 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：158
- pass_dates：156
- fail_dates：2
- eligible_train_rows：11038

## 2.1) 样本拒绝分布

- total_rows：11211
- learnable_rows：11038
- rejected_rows：173

| reason | count |
| --- | --- |
| pending_next_snapshot | 173 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11038
- pos/neg：1905/9133
- feature_coverage：1.0
- pass_trade_dates：156
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
