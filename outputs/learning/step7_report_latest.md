# Step7 自学习报告（latest）

- 生成时间：2026-08-26 20:01:01
- RunMode：auto_daily
- Today：20260826
- LatestSnapshot：20260826
- LabelUpperBound：20260826

## 1) 最新命中

- trade_date：20260825
- verify_date：20260826
- hit/topn：3/10
- hit_rate：0.3
- top1：0/1，hit_rate=0.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：3/10，hit_rate=0.3
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260825.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260812 | 20260813 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20260813 | 20260814 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260814 | 20260817 | 0.0 | 0.0 | 0.0 | 0.3 |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20260819 | 20260820 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260820 | 20260821 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260821 | 20260824 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260824 | 20260825 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260825 | 20260826 | 0.0 | 0.3333 | 0.2 | 0.3 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 141 | 141 | 71 | 0.5035 |
| Top3 | 141 | 423 | 182 | 0.4303 |
| Top5 | 141 | 705 | 267 | 0.3787 |
| Top10 | 141 | 1410 | 444 | 0.3149 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：143
- pass_dates：141
- fail_dates：2
- eligible_train_rows：10200

## 2.1) 样本拒绝分布

- total_rows：10342
- learnable_rows：10200
- rejected_rows：142

| reason | count |
| --- | --- |
| pending_next_snapshot | 142 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10200
- pos/neg：1716/8484
- feature_coverage：1.0
- pass_trade_dates：141
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
