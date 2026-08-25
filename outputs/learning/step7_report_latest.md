# Step7 自学习报告（latest）

- 生成时间：2026-08-25 19:20:17
- RunMode：auto_daily
- Today：20260825
- LatestSnapshot：20260825
- LabelUpperBound：20260825

## 1) 最新命中

- trade_date：20260824
- verify_date：20260825
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260824.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260811 | 20260812 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260812 | 20260813 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20260813 | 20260814 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260814 | 20260817 | 0.0 | 0.0 | 0.0 | 0.3 |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20260819 | 20260820 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260820 | 20260821 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260821 | 20260824 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260824 | 20260825 | 0.0 | 0.3333 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 140 | 140 | 71 | 0.5071 |
| Top3 | 140 | 420 | 181 | 0.431 |
| Top5 | 140 | 700 | 266 | 0.38 |
| Top10 | 140 | 1400 | 441 | 0.315 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：142
- pass_dates：140
- fail_dates：2
- eligible_train_rows：10139

## 2.1) 样本拒绝分布

- total_rows：10293
- learnable_rows：10139
- rejected_rows：154

| reason | count |
| --- | --- |
| pending_next_snapshot | 154 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10139
- pos/neg：1703/8436
- feature_coverage：1.0
- pass_trade_dates：140
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
