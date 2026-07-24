# Step7 自学习报告（latest）

- 生成时间：2026-07-24 20:31:48
- RunMode：auto_daily
- Today：20260724
- LatestSnapshot：20260724
- LabelUpperBound：20260724

## 1) 最新命中

- trade_date：20260723
- verify_date：20260724
- hit/topn：4/10
- hit_rate：0.4
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260723.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260710 | 20260713 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260713 | 20260714 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20260714 | 20260715 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260715 | 20260716 | 0.0 | 0.6667 | 0.6 | 0.4 |
| 20260716 | 20260717 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260717 | 20260720 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260720 | 20260721 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260721 | 20260722 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260722 | 20260723 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20260723 | 20260724 | 1.0 | 0.3333 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 118 | 118 | 63 | 0.5339 |
| Top3 | 118 | 354 | 160 | 0.452 |
| Top5 | 118 | 590 | 234 | 0.3966 |
| Top10 | 118 | 1180 | 386 | 0.3271 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：120
- pass_dates：118
- fail_dates：2
- eligible_train_rows：8600

## 2.1) 样本拒绝分布

- total_rows：8734
- learnable_rows：8600
- rejected_rows：134

| reason | count |
| --- | --- |
| pending_next_snapshot | 134 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8600
- pos/neg：1411/7189
- feature_coverage：1.0
- pass_trade_dates：118
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
