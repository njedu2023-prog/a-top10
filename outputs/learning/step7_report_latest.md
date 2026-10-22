# Step7 自学习报告（latest）

- 生成时间：2026-10-23 00:00:42
- RunMode：auto_daily
- Today：20261023
- LatestSnapshot：20261022
- LabelUpperBound：20261022

## 1) 最新命中

- trade_date：20261021
- verify_date：20261022
- hit/topn：5/10
- hit_rate：0.5
- top1：1/1，hit_rate=1.0
- top3：3/3，hit_rate=1.0
- top5：4/5，hit_rate=0.8
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261021.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261008 | 20261009 | 0.0 | 0.6667 | 0.4 | 0.4 |
| 20261009 | 20261012 | 1.0 | 0.6667 | 0.6 | 0.3 |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |
| 20261020 | 20261021 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261021 | 20261022 | 1.0 | 1.0 | 0.8 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 176 | 176 | 88 | 0.5 |
| Top3 | 176 | 528 | 232 | 0.4394 |
| Top5 | 176 | 880 | 334 | 0.3795 |
| Top10 | 176 | 1760 | 561 | 0.3187 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：178
- pass_dates：176
- fail_dates：2
- eligible_train_rows：12064

## 2.1) 样本拒绝分布

- total_rows：12187
- learnable_rows：12064
- rejected_rows：123

| reason | count |
| --- | --- |
| pending_next_snapshot | 123 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12064
- pos/neg：2145/9919
- feature_coverage：1.0
- pass_trade_dates：176
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
