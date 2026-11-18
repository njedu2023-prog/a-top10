# Step7 自学习报告（latest）

- 生成时间：2026-11-19 03:34:00
- RunMode：auto_daily
- Today：20261119
- LatestSnapshot：20261118
- LabelUpperBound：20261118

## 1) 最新命中

- trade_date：20261117
- verify_date：20261118
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261117.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261104 | 20261105 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20261105 | 20261106 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20261106 | 20261109 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20261109 | 20261110 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261110 | 20261111 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20261111 | 20261112 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20261112 | 20261113 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261113 | 20261116 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20261116 | 20261117 | 1.0 | 0.3333 | 0.2 | 0.3 |
| 20261117 | 20261118 | 0.0 | 0.3333 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 195 | 195 | 95 | 0.4872 |
| Top3 | 195 | 585 | 256 | 0.4376 |
| Top5 | 195 | 975 | 368 | 0.3774 |
| Top10 | 195 | 1950 | 619 | 0.3174 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：197
- pass_dates：195
- fail_dates：2
- eligible_train_rows：13081

## 2.1) 样本拒绝分布

- total_rows：13208
- learnable_rows：13081
- rejected_rows：127

| reason | count |
| --- | --- |
| pending_next_snapshot | 127 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：13081
- pos/neg：2367/10714
- feature_coverage：1.0
- pass_trade_dates：195
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
