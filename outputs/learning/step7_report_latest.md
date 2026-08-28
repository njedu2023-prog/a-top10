# Step7 自学习报告（latest）

- 生成时间：2026-08-28 19:36:35
- RunMode：auto_daily
- Today：20260828
- LatestSnapshot：20260828
- LabelUpperBound：20260828

## 1) 最新命中

- trade_date：20260827
- verify_date：20260828
- hit/topn：5/10
- hit_rate：0.5
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260827.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260814 | 20260817 | 0.0 | 0.0 | 0.0 | 0.3 |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20260819 | 20260820 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260820 | 20260821 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260821 | 20260824 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260824 | 20260825 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260825 | 20260826 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260826 | 20260827 | 1.0 | 0.6667 | 0.8 | 0.4 |
| 20260827 | 20260828 | 1.0 | 0.3333 | 0.4 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 143 | 143 | 73 | 0.5105 |
| Top3 | 143 | 429 | 185 | 0.4312 |
| Top5 | 143 | 715 | 273 | 0.3818 |
| Top10 | 143 | 1430 | 453 | 0.3168 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：145
- pass_dates：143
- fail_dates：2
- eligible_train_rows：10317

## 2.1) 样本拒绝分布

- total_rows：10488
- learnable_rows：10317
- rejected_rows：171

| reason | count |
| --- | --- |
| pending_next_snapshot | 171 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10317
- pos/neg：1749/8568
- feature_coverage：1.0
- pass_trade_dates：143
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
