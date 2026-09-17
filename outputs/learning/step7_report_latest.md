# Step7 自学习报告（latest）

- 生成时间：2026-09-17 23:39:57
- RunMode：auto_daily
- Today：20260917
- LatestSnapshot：20260917
- LabelUpperBound：20260917

## 1) 最新命中

- trade_date：20260916
- verify_date：20260917
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：0/5，hit_rate=0.0
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260916.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260903 | 20260904 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260904 | 20260907 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260907 | 20260908 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260908 | 20260909 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260909 | 20260910 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260910 | 20260911 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260911 | 20260914 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20260914 | 20260915 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20260915 | 20260916 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260916 | 20260917 | 0.0 | 0.0 | 0.0 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 157 | 157 | 76 | 0.4841 |
| Top3 | 157 | 471 | 199 | 0.4225 |
| Top5 | 157 | 785 | 292 | 0.372 |
| Top10 | 157 | 1570 | 494 | 0.3146 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：159
- pass_dates：157
- fail_dates：2
- eligible_train_rows：11118

## 2.1) 样本拒绝分布

- total_rows：11255
- learnable_rows：11118
- rejected_rows：137

| reason | count |
| --- | --- |
| pending_next_snapshot | 137 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11118
- pos/neg：1914/9204
- feature_coverage：1.0
- pass_trade_dates：157
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
