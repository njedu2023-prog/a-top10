# Step7 自学习报告（latest）

- 生成时间：2026-10-23 23:53:07
- RunMode：auto_daily
- Today：20261023
- LatestSnapshot：20261023
- LabelUpperBound：20261023

## 1) 最新命中

- trade_date：20261022
- verify_date：20261023
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261022.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261009 | 20261012 | 1.0 | 0.6667 | 0.6 | 0.3 |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |
| 20261020 | 20261021 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261021 | 20261022 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20261022 | 20261023 | 0.0 | 0.3333 | 0.2 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 177 | 177 | 88 | 0.4972 |
| Top3 | 177 | 531 | 233 | 0.4388 |
| Top5 | 177 | 885 | 335 | 0.3785 |
| Top10 | 177 | 1770 | 565 | 0.3192 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：179
- pass_dates：177
- fail_dates：2
- eligible_train_rows：12094

## 2.1) 样本拒绝分布

- total_rows：12251
- learnable_rows：12094
- rejected_rows：157

| reason | count |
| --- | --- |
| pending_next_snapshot | 157 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12094
- pos/neg：2156/9938
- feature_coverage：1.0
- pass_trade_dates：177
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
