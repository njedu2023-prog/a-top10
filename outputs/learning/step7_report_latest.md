# Step7 自学习报告（latest）

- 生成时间：2026-11-05 19:18:36
- RunMode：auto_daily
- Today：20261105
- LatestSnapshot：20261105
- LabelUpperBound：20261105

## 1) 最新命中

- trade_date：20261104
- verify_date：20261105
- hit/topn：2/10
- hit_rate：0.2
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20261104.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20261022 | 20261023 | 0.0 | 0.3333 | 0.2 | 0.4 |
| 20261023 | 20261026 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20261026 | 20261027 | 0.0 | 0.3333 | 0.6 | 0.5 |
| 20261027 | 20261028 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20261028 | 20261029 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20261029 | 20261030 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20261030 | 20261102 | 1.0 | 0.3333 | 0.2 | 0.3 |
| 20261102 | 20261103 | 1.0 | 0.6667 | 0.4 | 0.2 |
| 20261103 | 20261104 | 1.0 | 0.3333 | 0.6 | 0.4 |
| 20261104 | 20261105 | 0.0 | 0.3333 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 186 | 186 | 91 | 0.4892 |
| Top3 | 186 | 558 | 242 | 0.4337 |
| Top5 | 186 | 930 | 351 | 0.3774 |
| Top10 | 186 | 1860 | 591 | 0.3177 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：188
- pass_dates：186
- fail_dates：2
- eligible_train_rows：12635

## 2.1) 样本拒绝分布

- total_rows：12799
- learnable_rows：12635
- rejected_rows：164

| reason | count |
| --- | --- |
| pending_next_snapshot | 164 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：12635
- pos/neg：2267/10368
- feature_coverage：1.0
- pass_trade_dates：186
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
