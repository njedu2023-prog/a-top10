# Step7 自学习报告（latest）

- 生成时间：2026-07-11 20:52:25
- RunMode：auto_daily
- Today：20260711
- LatestSnapshot：20260710
- LabelUpperBound：20260710

## 1) 最新命中

- trade_date：20260709
- verify_date：20260710
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260709.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260626 | 20260629 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20260629 | 20260630 | 1.0 | 0.3333 | 0.4 | 0.2 |
| 20260630 | 20260701 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260701 | 20260702 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260702 | 20260703 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260703 | 20260706 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260706 | 20260707 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260707 | 20260708 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260708 | 20260709 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260709 | 20260710 | 0.0 | 0.3333 | 0.2 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 108 | 108 | 59 | 0.5463 |
| Top3 | 108 | 324 | 149 | 0.4599 |
| Top5 | 108 | 540 | 219 | 0.4056 |
| Top10 | 108 | 1080 | 362 | 0.3352 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：110
- pass_dates：108
- fail_dates：2
- eligible_train_rows：7917

## 2.1) 样本拒绝分布

- total_rows：8101
- learnable_rows：7917
- rejected_rows：184

| reason | count |
| --- | --- |
| pending_next_snapshot | 184 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：7917
- pos/neg：1308/6609
- feature_coverage：1.0
- pass_trade_dates：108
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
