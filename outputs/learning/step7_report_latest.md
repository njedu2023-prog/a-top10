# Step7 自学习报告（latest）

- 生成时间：2026-07-20 21:19:15
- RunMode：auto_daily
- Today：20260720
- LatestSnapshot：20260720
- LabelUpperBound：20260720

## 1) 最新命中

- trade_date：20260717
- verify_date：20260720
- hit/topn：3/10
- hit_rate：0.3
- top1：0/1，hit_rate=0.0
- top3：0/3，hit_rate=0.0
- top5：1/5，hit_rate=0.2
- top10：3/10，hit_rate=0.3
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260717.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260706 | 20260707 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260707 | 20260708 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260708 | 20260709 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260709 | 20260710 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260710 | 20260713 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260713 | 20260714 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20260714 | 20260715 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260715 | 20260716 | 0.0 | 0.6667 | 0.6 | 0.4 |
| 20260716 | 20260717 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260717 | 20260720 | 0.0 | 0.0 | 0.2 | 0.3 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 114 | 114 | 60 | 0.5263 |
| Top3 | 114 | 342 | 154 | 0.4503 |
| Top5 | 114 | 570 | 226 | 0.3965 |
| Top10 | 114 | 1140 | 374 | 0.3281 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：116
- pass_dates：114
- fail_dates：2
- eligible_train_rows：8262

## 2.1) 样本拒绝分布

- total_rows：8409
- learnable_rows：8262
- rejected_rows：147

| reason | count |
| --- | --- |
| pending_next_snapshot | 147 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8262
- pos/neg：1361/6901
- feature_coverage：1.0
- pass_trade_dates：114
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
