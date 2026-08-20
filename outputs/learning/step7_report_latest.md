# Step7 自学习报告（latest）

- 生成时间：2026-08-20 19:19:36
- RunMode：auto_daily
- Today：20260820
- LatestSnapshot：20260820
- LabelUpperBound：20260820

## 1) 最新命中

- trade_date：20260819
- verify_date：20260820
- hit/topn：2/10
- hit_rate：0.2
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：2/10，hit_rate=0.2
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260819.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260806 | 20260807 | 1.0 | 0.3333 | 0.2 | 0.4 |
| 20260807 | 20260810 | 1.0 | 0.3333 | 0.4 | 0.2 |
| 20260810 | 20260811 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260811 | 20260812 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260812 | 20260813 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20260813 | 20260814 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260814 | 20260817 | 0.0 | 0.0 | 0.0 | 0.3 |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |
| 20260819 | 20260820 | 1.0 | 0.3333 | 0.2 | 0.2 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 137 | 137 | 71 | 0.5182 |
| Top3 | 137 | 411 | 180 | 0.438 |
| Top5 | 137 | 685 | 262 | 0.3825 |
| Top10 | 137 | 1370 | 433 | 0.3161 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：139
- pass_dates：137
- fail_dates：2
- eligible_train_rows：9989

## 2.1) 样本拒绝分布

- total_rows：10141
- learnable_rows：9989
- rejected_rows：152

| reason | count |
| --- | --- |
| pending_next_snapshot | 152 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9989
- pos/neg：1676/8313
- feature_coverage：1.0
- pass_trade_dates：137
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
