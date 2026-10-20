# Step7 自学习报告（latest）

- 生成时间：2026-10-20 23:50:59
- RunMode：auto_daily
- Today：20261020
- LatestSnapshot：20261020
- LabelUpperBound：20261020

## 1) 最新命中

- trade_date：20261019
- verify_date：20261020
- hit/topn：5/10
- hit_rate：0.5
- top1：1/1，hit_rate=1.0
- top3：3/3，hit_rate=1.0
- top5：3/5，hit_rate=0.6
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261019.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260929 | 20260930 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20260930 | 20261008 | 1.0 | 0.6667 | 0.6 | 0.6 |
| 20261008 | 20261009 | 0.0 | 0.6667 | 0.4 | 0.4 |
| 20261009 | 20261012 | 1.0 | 0.6667 | 0.6 | 0.3 |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 174 | 174 | 86 | 0.4943 |
| Top3 | 174 | 522 | 227 | 0.4349 |
| Top5 | 174 | 870 | 327 | 0.3759 |
| Top10 | 174 | 1740 | 551 | 0.3167 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：176
- pass_dates：174
- fail_dates：2
- eligible_train_rows：11974

## 2.1) 样本拒绝分布

- total_rows：12113
- learnable_rows：11974
- rejected_rows：139

| reason | count |
| --- | --- |
| pending_next_snapshot | 139 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11974
- pos/neg：2120/9854
- feature_coverage：1.0
- pass_trade_dates：174
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
