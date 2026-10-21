# Step7 自学习报告（latest）

- 生成时间：2026-10-21 19:15:42
- RunMode：auto_daily
- Today：20261021
- LatestSnapshot：20261021
- LabelUpperBound：20261021

## 1) 最新命中

- trade_date：20261020
- verify_date：20261021
- hit/topn：5/10
- hit_rate：0.5
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：3/5，hit_rate=0.6
- top10：5/10，hit_rate=0.5
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261020.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260930 | 20261008 | 1.0 | 0.6667 | 0.6 | 0.6 |
| 20261008 | 20261009 | 0.0 | 0.6667 | 0.4 | 0.4 |
| 20261009 | 20261012 | 1.0 | 0.6667 | 0.6 | 0.3 |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |
| 20261020 | 20261021 | 1.0 | 0.6667 | 0.6 | 0.5 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 175 | 175 | 87 | 0.4971 |
| Top3 | 175 | 525 | 229 | 0.4362 |
| Top5 | 175 | 875 | 330 | 0.3771 |
| Top10 | 175 | 1750 | 556 | 0.3177 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：177
- pass_dates：175
- fail_dates：2
- eligible_train_rows：12020

## 2.1) 样本拒绝分布

- total_rows：12157
- learnable_rows：12020
- rejected_rows：137

| reason | count |
| --- | --- |
| pending_next_snapshot | 137 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12020
- pos/neg：2136/9884
- feature_coverage：1.0
- pass_trade_dates：175
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
