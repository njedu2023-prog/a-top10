# Step7 自学习报告（latest）

- 生成时间：2026-08-03 19:42:16
- RunMode：auto_daily
- Today：20260803
- LatestSnapshot：20260803
- LabelUpperBound：20260803

## 1) 最新命中

- trade_date：20260731
- verify_date：20260803
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：0/5，hit_rate=0.0
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260731.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260720 | 20260721 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260721 | 20260722 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260722 | 20260723 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20260723 | 20260724 | 1.0 | 0.3333 | 0.4 | 0.4 |
| 20260724 | 20260727 | 1.0 | 0.6667 | 0.8 | 0.5 |
| 20260727 | 20260728 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260728 | 20260729 | 0.0 | 0.0 | 0.2 | 0.2 |
| 20260729 | 20260730 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260730 | 20260731 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260731 | 20260803 | 0.0 | 0.0 | 0.0 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 124 | 124 | 64 | 0.5161 |
| Top3 | 124 | 372 | 164 | 0.4409 |
| Top5 | 124 | 620 | 242 | 0.3903 |
| Top10 | 124 | 1240 | 399 | 0.3218 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：126
- pass_dates：124
- fail_dates：2
- eligible_train_rows：9046

## 2.1) 样本拒绝分布

- total_rows：9208
- learnable_rows：9046
- rejected_rows：162

| reason | count |
| --- | --- |
| pending_next_snapshot | 162 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9046
- pos/neg：1483/7563
- feature_coverage：1.0
- pass_trade_dates：124
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
