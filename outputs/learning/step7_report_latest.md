# Step7 自学习报告（latest）

- 生成时间：2026-09-29 20:51:29
- RunMode：auto_daily
- Today：20260929
- LatestSnapshot：20260929
- LabelUpperBound：20260929

## 1) 最新命中

- trade_date：20260928
- verify_date：20260929
- hit/topn：2/10
- hit_rate：0.2
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260928.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260914 | 20260915 | 0.0 | 0.6667 | 0.4 | 0.2 |
| 20260915 | 20260916 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260916 | 20260917 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260917 | 20260918 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20260918 | 20260921 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260921 | 20260922 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260922 | 20260923 | 0.0 | 0.6667 | 0.6 | 0.5 |
| 20260923 | 20260924 | 1.0 | 0.6667 | 0.6 | 0.4 |
| 20260924 | 20260928 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260928 | 20260929 | 1.0 | 0.3333 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 164 | 164 | 80 | 0.4878 |
| Top3 | 164 | 492 | 209 | 0.4248 |
| Top5 | 164 | 820 | 306 | 0.3732 |
| Top10 | 164 | 1640 | 515 | 0.314 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：166
- pass_dates：164
- fail_dates：2
- eligible_train_rows：11510

## 2.1) 样本拒绝分布

- total_rows：11654
- learnable_rows：11510
- rejected_rows：144

| reason | count |
| --- | --- |
| pending_next_snapshot | 144 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：11510
- pos/neg：2011/9499
- feature_coverage：1.0
- pass_trade_dates：164
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
