# Step7 自学习报告（latest）

- 生成时间：2026-07-17 20:25:08
- RunMode：auto_daily
- Today：20260717
- LatestSnapshot：20260717
- LabelUpperBound：20260717

## 1) 最新命中

- trade_date：20260716
- verify_date：20260717
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260716.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260703 | 20260706 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260706 | 20260707 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260707 | 20260708 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260708 | 20260709 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260709 | 20260710 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260710 | 20260713 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260713 | 20260714 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20260714 | 20260715 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260715 | 20260716 | 0.0 | 0.6667 | 0.6 | 0.4 |
| 20260716 | 20260717 | 0.0 | 0.3333 | 0.2 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 113 | 113 | 60 | 0.531 |
| Top3 | 113 | 339 | 154 | 0.4543 |
| Top5 | 113 | 565 | 225 | 0.3982 |
| Top10 | 113 | 1130 | 371 | 0.3283 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：115
- pass_dates：113
- fail_dates：2
- eligible_train_rows：8230

## 2.1) 样本拒绝分布

- total_rows：8355
- learnable_rows：8230
- rejected_rows：125

| reason | count |
| --- | --- |
| pending_next_snapshot | 125 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8230
- pos/neg：1353/6877
- feature_coverage：1.0
- pass_trade_dates：113
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
