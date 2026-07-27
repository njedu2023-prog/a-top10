# Step7 自学习报告（latest）

- 生成时间：2026-07-27 21:54:06
- RunMode：auto_daily
- Today：20260727
- LatestSnapshot：20260727
- LabelUpperBound：20260727

## 1) 最新命中

- trade_date：20260724
- verify_date：20260727
- hit/topn：5/10
- hit_rate：0.5
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：4/5，hit_rate=0.8
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260724.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260713 | 20260714 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20260714 | 20260715 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260715 | 20260716 | 0.0 | 0.6667 | 0.6 | 0.4 |
| 20260716 | 20260717 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260717 | 20260720 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260720 | 20260721 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260721 | 20260722 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260722 | 20260723 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20260723 | 20260724 | 1.0 | 0.3333 | 0.4 | 0.4 |
| 20260724 | 20260727 | 1.0 | 0.6667 | 0.8 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 119 | 119 | 64 | 0.5378 |
| Top3 | 119 | 357 | 162 | 0.4538 |
| Top5 | 119 | 595 | 238 | 0.4 |
| Top10 | 119 | 1190 | 391 | 0.3286 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：121
- pass_dates：119
- fail_dates：2
- eligible_train_rows：8641

## 2.1) 样本拒绝分布

- total_rows：8845
- learnable_rows：8641
- rejected_rows：204

| reason | count |
| --- | --- |
| pending_next_snapshot | 204 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8641
- pos/neg：1419/7222
- feature_coverage：1.0
- pass_trade_dates：119
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
