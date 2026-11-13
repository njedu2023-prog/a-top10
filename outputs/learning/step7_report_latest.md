# Step7 自学习报告（latest）

- 生成时间：2026-11-13 22:33:25
- RunMode：auto_daily
- Today：20261113
- LatestSnapshot：20261113
- LabelUpperBound：20261113

## 1) 最新命中

- trade_date：20261112
- verify_date：20261113
- hit/topn：1/10
- hit_rate：0.1
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261112.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261030 | 20261102 | 1.0 | 0.3333 | 0.2 | 0.3 |
| 20261102 | 20261103 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20261103 | 20261104 | 1.0 | 0.3333 | 0.6 | 0.4 |
| 20261104 | 20261105 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20261105 | 20261106 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20261106 | 20261109 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20261109 | 20261110 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261110 | 20261111 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20261111 | 20261112 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20261112 | 20261113 | 0.0 | 0.3333 | 0.2 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 192 | 192 | 94 | 0.4896 |
| Top3 | 192 | 576 | 252 | 0.4375 |
| Top5 | 192 | 960 | 363 | 0.3781 |
| Top10 | 192 | 1920 | 610 | 0.3177 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：194
- pass_dates：192
- fail_dates：2
- eligible_train_rows：12983

## 2.1) 样本拒绝分布

- total_rows：13111
- learnable_rows：12983
- rejected_rows：128

| reason | count |
| --- | --- |
| pending_next_snapshot | 128 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12983
- pos/neg：2343/10640
- feature_coverage：1.0
- pass_trade_dates：192
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
