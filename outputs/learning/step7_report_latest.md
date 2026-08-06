# Step7 自学习报告（latest）

- 生成时间：2026-08-06 21:13:34
- RunMode：auto_daily
- Today：20260806
- LatestSnapshot：20260806
- LabelUpperBound：20260806

## 1) 最新命中

- trade_date：20260805
- verify_date：20260806
- hit/topn：2/10
- hit_rate：0.2
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260805.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260723 | 20260724 | 1.0 | 0.3333 | 0.4 | 0.4 |
| 20260724 | 20260727 | 1.0 | 0.6667 | 0.8 | 0.5 |
| 20260727 | 20260728 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260728 | 20260729 | 0.0 | 0.0 | 0.2 | 0.2 |
| 20260729 | 20260730 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260730 | 20260731 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260731 | 20260803 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260803 | 20260804 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260804 | 20260805 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260805 | 20260806 | 0.0 | 0.3333 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 127 | 127 | 65 | 0.5118 |
| Top3 | 127 | 381 | 169 | 0.4436 |
| Top5 | 127 | 635 | 249 | 0.3921 |
| Top10 | 127 | 1270 | 409 | 0.322 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：129
- pass_dates：127
- fail_dates：2
- eligible_train_rows：9332

## 2.1) 样本拒绝分布

- total_rows：9495
- learnable_rows：9332
- rejected_rows：163

| reason | count |
| --- | --- |
| pending_next_snapshot | 163 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9332
- pos/neg：1552/7780
- feature_coverage：1.0
- pass_trade_dates：127
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
