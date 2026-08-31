# Step7 自学习报告（latest）

- 生成时间：2026-08-31 19:21:49
- RunMode：auto_daily
- Today：20260831
- LatestSnapshot：20260831
- LabelUpperBound：20260831

## 1) 最新命中

- trade_date：20260828
- verify_date：20260831
- hit/topn：2/10
- hit_rate：0.2
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260828.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20260819 | 20260820 | 1.0 | 0.3333 | 0.2 | 0.2 |
| 20260820 | 20260821 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260821 | 20260824 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260824 | 20260825 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260825 | 20260826 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260826 | 20260827 | 1.0 | 0.6667 | 0.8 | 0.4 |
| 20260827 | 20260828 | 1.0 | 0.3333 | 0.4 | 0.5 |
| 20260828 | 20260831 | 1.0 | 0.3333 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 144 | 144 | 74 | 0.5139 |
| Top3 | 144 | 432 | 186 | 0.4306 |
| Top5 | 144 | 720 | 274 | 0.3806 |
| Top10 | 144 | 1440 | 455 | 0.316 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：146
- pass_dates：144
- fail_dates：2
- eligible_train_rows：10395

## 2.1) 样本拒绝分布

- total_rows：10567
- learnable_rows：10395
- rejected_rows：172

| reason | count |
| --- | --- |
| pending_next_snapshot | 172 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：10395
- pos/neg：1767/8628
- feature_coverage：1.0
- pass_trade_dates：144
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
