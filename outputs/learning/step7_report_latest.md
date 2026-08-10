# Step7 自学习报告（latest）

- 生成时间：2026-08-10 20:13:25
- RunMode：auto_daily
- Today：20260810
- LatestSnapshot：20260810
- LabelUpperBound：20260810

## 1) 最新命中

- trade_date：20260807
- verify_date：20260810
- hit/topn：2/10
- hit_rate：0.2
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260807.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260727 | 20260728 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260728 | 20260729 | 0.0 | 0.0 | 0.2 | 0.2 |
| 20260729 | 20260730 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260730 | 20260731 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260731 | 20260803 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260803 | 20260804 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260804 | 20260805 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260805 | 20260806 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260806 | 20260807 | 1.0 | 0.3333 | 0.2 | 0.4 |
| 20260807 | 20260810 | 1.0 | 0.3333 | 0.4 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 129 | 129 | 67 | 0.5194 |
| Top3 | 129 | 387 | 171 | 0.4419 |
| Top5 | 129 | 645 | 252 | 0.3907 |
| Top10 | 129 | 1290 | 415 | 0.3217 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：131
- pass_dates：129
- fail_dates：2
- eligible_train_rows：9461

## 2.1) 样本拒绝分布

- total_rows：9648
- learnable_rows：9461
- rejected_rows：187

| reason | count |
| --- | --- |
| pending_next_snapshot | 187 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9461
- pos/neg：1576/7885
- feature_coverage：1.0
- pass_trade_dates：129
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
