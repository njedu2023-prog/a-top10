# Step7 自学习报告（latest）

- 生成时间：2026-10-30 01:31:55
- RunMode：auto_daily
- Today：20261030
- LatestSnapshot：20261029
- LabelUpperBound：20261029

## 1) 最新命中

- trade_date：20261028
- verify_date：20261029
- hit/topn：2/10
- hit_rate：0.2
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：2/5，hit_rate=0.4
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261028.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261015 | 20261016 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20261016 | 20261019 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261019 | 20261020 | 1.0 | 1.0 | 0.6 | 0.5 |
| 20261020 | 20261021 | 1.0 | 0.6667 | 0.6 | 0.5 |
| 20261021 | 20261022 | 1.0 | 1.0 | 0.8 | 0.5 |
| 20261022 | 20261023 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261023 | 20261026 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20261026 | 20261027 | 0.0 | 0.3333 | 0.6 | 0.5 |
| 20261027 | 20261028 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20261028 | 20261029 | 0.0 | 0.3333 | 0.4 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 181 | 181 | 88 | 0.4862 |
| Top3 | 181 | 543 | 237 | 0.4365 |
| Top5 | 181 | 905 | 343 | 0.379 |
| Top10 | 181 | 1810 | 577 | 0.3188 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：183
- pass_dates：181
- fail_dates：2
- eligible_train_rows：12302

## 2.1) 样本拒绝分布

- total_rows：12433
- learnable_rows：12302
- rejected_rows：131

| reason | count |
| --- | --- |
| pending_next_snapshot | 131 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12302
- pos/neg：2200/10102
- feature_coverage：1.0
- pass_trade_dates：181
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
