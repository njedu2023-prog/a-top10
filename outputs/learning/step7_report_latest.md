# Step7 自学习报告（latest）

- 生成时间：2026-07-22 21:07:04
- RunMode：auto_daily
- Today：20260722
- LatestSnapshot：20260722
- LabelUpperBound：20260722

## 1) 最新命中

- trade_date：20260721
- verify_date：20260722
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：0/5，hit_rate=0.0
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260721.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260708 | 20260709 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260709 | 20260710 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260710 | 20260713 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260713 | 20260714 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20260714 | 20260715 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260715 | 20260716 | 0.0 | 0.6667 | 0.6 | 0.4 |
| 20260716 | 20260717 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260717 | 20260720 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260720 | 20260721 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260721 | 20260722 | 0.0 | 0.0 | 0.0 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 116 | 116 | 61 | 0.5259 |
| Top3 | 116 | 348 | 156 | 0.4483 |
| Top5 | 116 | 580 | 228 | 0.3931 |
| Top10 | 116 | 1160 | 377 | 0.325 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：118
- pass_dates：116
- fail_dates：2
- eligible_train_rows：8437

## 2.1) 样本拒绝分布

- total_rows：8576
- learnable_rows：8437
- rejected_rows：139

| reason | count |
| --- | --- |
| pending_next_snapshot | 139 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8437
- pos/neg：1378/7059
- feature_coverage：1.0
- pass_trade_dates：116
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
