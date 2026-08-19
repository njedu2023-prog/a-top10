# Step7 自学习报告（latest）

- 生成时间：2026-08-19 19:37:19
- RunMode：auto_daily
- Today：20260819
- LatestSnapshot：20260819
- LabelUpperBound：20260819

## 1) 最新命中

- trade_date：20260818
- verify_date：20260819
- hit/topn：1/10
- hit_rate：0.1
- top1：1/1，hit_rate=1.0
- top3：1/3，hit_rate=0.3333
- top5：1/5，hit_rate=0.2
- top10：1/10，hit_rate=0.1
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260818.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260805 | 20260806 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260806 | 20260807 | 1.0 | 0.3333 | 0.2 | 0.4 |
| 20260807 | 20260810 | 1.0 | 0.3333 | 0.4 | 0.2 |
| 20260810 | 20260811 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260811 | 20260812 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260812 | 20260813 | 1.0 | 1.0 | 0.6 | 0.3 |
| 20260813 | 20260814 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260814 | 20260817 | 0.0 | 0.0 | 0.0 | 0.3 |
| 20260817 | 20260818 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260818 | 20260819 | 1.0 | 0.3333 | 0.2 | 0.1 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 136 | 136 | 70 | 0.5147 |
| Top3 | 136 | 408 | 179 | 0.4387 |
| Top5 | 136 | 680 | 261 | 0.3838 |
| Top10 | 136 | 1360 | 431 | 0.3169 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：138
- pass_dates：136
- fail_dates：2
- eligible_train_rows：9957

## 2.1) 样本拒绝分布

- total_rows：10082
- learnable_rows：9957
- rejected_rows：125

| reason | count |
| --- | --- |
| pending_next_snapshot | 125 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9957
- pos/neg：1672/8285
- feature_coverage：1.0
- pass_trade_dates：136
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
