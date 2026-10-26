# Step7 自学习报告（latest）

- 生成时间：2026-10-27 01:37:48
- RunMode：auto_daily
- Today：20261027
- LatestSnapshot：20261026
- LabelUpperBound：20261026

## 1) 最新命中

- trade_date：20261023
- verify_date：20261026
- hit/topn：3/10
- hit_rate：0.3
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：3/10，hit_rate=0.3
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261023.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261012 | 20261013 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261013 | 20261014 | 1.0 | 0.6667 | 0.4 | 0.4 |
| 20261014 | 20261015 | 0.0 | 0.3333 | 0.4 | 0.3 |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |
| 20261020 | 20261021 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261021 | 20261022 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20261022 | 20261023 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261023 | 20261026 | 0.0 | 0.3333 | 0.2 | 0.3 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 178 | 178 | 88 | 0.4944 |
| Top3 | 178 | 534 | 234 | 0.4382 |
| Top5 | 178 | 890 | 336 | 0.3775 |
| Top10 | 178 | 1780 | 568 | 0.3191 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：180
- pass_dates：178
- fail_dates：2
- eligible_train_rows：12158

## 2.1) 样本拒绝分布

- total_rows：12298
- learnable_rows：12158
- rejected_rows：140

| reason | count |
| --- | --- |
| pending_next_snapshot | 140 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12158
- pos/neg：2170/9988
- feature_coverage：1.0
- pass_trade_dates：178
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
