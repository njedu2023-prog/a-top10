# Step7 自学习报告（latest）

- 生成时间：2026-08-11 19:54:54
- RunMode：auto_daily
- Today：20260811
- LatestSnapshot：20260811
- LabelUpperBound：20260811

## 1) 最新命中

- trade_date：20260810
- verify_date：20260811
- hit/topn：2/10
- hit_rate：0.2
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260810.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260728 | 20260729 | 0.0 | 0.0 | 0.2 | 0.2 |
| 20260729 | 20260730 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260730 | 20260731 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260731 | 20260803 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260803 | 20260804 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260804 | 20260805 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260805 | 20260806 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260806 | 20260807 | 1.0 | 0.3333 | 0.2 | 0.4 |
| 20260807 | 20260810 | 1.0 | 0.3333 | 0.4 | 0.2 |
| 20260810 | 20260811 | 0.0 | 0.3333 | 0.4 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 130 | 130 | 67 | 0.5154 |
| Top3 | 130 | 390 | 172 | 0.441 |
| Top5 | 130 | 650 | 254 | 0.3908 |
| Top10 | 130 | 1300 | 417 | 0.3208 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：132
- pass_dates：130
- fail_dates：2
- eligible_train_rows：9555

## 2.1) 样本拒绝分布

- total_rows：9701
- learnable_rows：9555
- rejected_rows：146

| reason | count |
| --- | --- |
| pending_next_snapshot | 146 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9555
- pos/neg：1591/7964
- feature_coverage：1.0
- pass_trade_dates：130
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
