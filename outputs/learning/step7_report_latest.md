# Step7 自学习报告（latest）

- 生成时间：2026-08-27 19:27:26
- RunMode：auto_daily
- Today：20260827
- LatestSnapshot：20260827
- LabelUpperBound：20260827

## 1) 最新命中

- trade_date：20260826
- verify_date：20260827
- hit/topn：4/10
- hit_rate：0.4
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：4/5，hit_rate=0.8
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260826.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260813 | 20260814 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260814 | 20260817 | 0.0 | 0.0 | 0.0 | 0.3 |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20260819 | 20260820 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260820 | 20260821 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260821 | 20260824 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260824 | 20260825 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260825 | 20260826 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260826 | 20260827 | 1.0 | 0.6667 | 0.8 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 142 | 142 | 72 | 0.507 |
| Top3 | 142 | 426 | 184 | 0.4319 |
| Top5 | 142 | 710 | 271 | 0.3817 |
| Top10 | 142 | 1420 | 448 | 0.3155 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：144
- pass_dates：142
- fail_dates：2
- eligible_train_rows：10249

## 2.1) 样本拒绝分布

- total_rows：10410
- learnable_rows：10249
- rejected_rows：161

| reason | count |
| --- | --- |
| pending_next_snapshot | 161 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10249
- pos/neg：1732/8517
- feature_coverage：1.0
- pass_trade_dates：142
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
