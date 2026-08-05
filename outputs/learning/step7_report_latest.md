# Step7 自学习报告（latest）

- 生成时间：2026-08-05 19:15:26
- RunMode：auto_daily
- Today：20260805
- LatestSnapshot：20260805
- LabelUpperBound：20260805

## 1) 最新命中

- trade_date：20260804
- verify_date：20260805
- hit/topn：3/10
- hit_rate：0.3
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：2/5，hit_rate=0.4
- top10：3/10，hit_rate=0.3
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260804.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260722 | 20260723 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20260723 | 20260724 | 1.0 | 0.3333 | 0.4 | 0.4 |
| 20260724 | 20260727 | 1.0 | 0.6667 | 0.8 | 0.5 |
| 20260727 | 20260728 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260728 | 20260729 | 0.0 | 0.0 | 0.2 | 0.2 |
| 20260729 | 20260730 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260730 | 20260731 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260731 | 20260803 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260803 | 20260804 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260804 | 20260805 | 1.0 | 0.6667 | 0.4 | 0.3 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 126 | 126 | 65 | 0.5159 |
| Top3 | 126 | 378 | 168 | 0.4444 |
| Top5 | 126 | 630 | 248 | 0.3937 |
| Top10 | 126 | 1260 | 407 | 0.323 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：128
- pass_dates：126
- fail_dates：2
- eligible_train_rows：9238

## 2.1) 样本拒绝分布

- total_rows：9425
- learnable_rows：9238
- rejected_rows：187

| reason | count |
| --- | --- |
| pending_next_snapshot | 187 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9238
- pos/neg：1533/7705
- feature_coverage：1.0
- pass_trade_dates：126
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
