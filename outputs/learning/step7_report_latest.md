# Step7 自学习报告（latest）

- 生成时间：2026-11-10 05:39:48
- RunMode：auto_daily
- Today：20261110
- LatestSnapshot：20261109
- LabelUpperBound：20261109

## 1) 最新命中

- trade_date：20261106
- verify_date：20261109
- hit/topn：4/10
- hit_rate：0.4
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261106.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261026 | 20261027 | 0.0 | 0.3333 | 0.6 | 0.5 |
| 20261027 | 20261028 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20261028 | 20261029 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20261029 | 20261030 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20261030 | 20261102 | 1.0 | 0.3333 | 0.2 | 0.3 |
| 20261102 | 20261103 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20261103 | 20261104 | 1.0 | 0.3333 | 0.6 | 0.4 |
| 20261104 | 20261105 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20261105 | 20261106 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20261106 | 20261109 | 0.0 | 0.3333 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 188 | 188 | 91 | 0.484 |
| Top3 | 188 | 564 | 244 | 0.4326 |
| Top5 | 188 | 940 | 354 | 0.3766 |
| Top10 | 188 | 1880 | 598 | 0.3181 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：190
- pass_dates：188
- fail_dates：2
- eligible_train_rows：12735

## 2.1) 样本拒绝分布

- total_rows：12879
- learnable_rows：12735
- rejected_rows：144

| reason | count |
| --- | --- |
| pending_next_snapshot | 144 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12735
- pos/neg：2287/10448
- feature_coverage：1.0
- pass_trade_dates：188
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
