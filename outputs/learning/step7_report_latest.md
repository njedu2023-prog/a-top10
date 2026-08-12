# Step7 自学习报告（latest）

- 生成时间：2026-08-12 19:17:39
- RunMode：auto_daily
- Today：20260812
- LatestSnapshot：20260812
- LabelUpperBound：20260812

## 1) 最新命中

- trade_date：20260811
- verify_date：20260812
- hit/topn：3/10
- hit_rate：0.3
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：2/5，hit_rate=0.4
- top10：3/10，hit_rate=0.3
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260811.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260729 | 20260730 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260730 | 20260731 | 0.0 | 0.3333 | 0.2 | 0.3 |
| 20260731 | 20260803 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260803 | 20260804 | 0.0 | 0.3333 | 0.4 | 0.4 |
| 20260804 | 20260805 | 1.0 | 0.6667 | 0.4 | 0.3 |
| 20260805 | 20260806 | 0.0 | 0.3333 | 0.2 | 0.2 |
| 20260806 | 20260807 | 1.0 | 0.3333 | 0.2 | 0.4 |
| 20260807 | 20260810 | 1.0 | 0.3333 | 0.4 | 0.2 |
| 20260810 | 20260811 | 0.0 | 0.3333 | 0.4 | 0.2 |
| 20260811 | 20260812 | 1.0 | 0.6667 | 0.4 | 0.3 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 131 | 131 | 68 | 0.5191 |
| Top3 | 131 | 393 | 174 | 0.4427 |
| Top5 | 131 | 655 | 256 | 0.3908 |
| Top10 | 131 | 1310 | 420 | 0.3206 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：133
- pass_dates：131
- fail_dates：2
- eligible_train_rows：9608

## 2.1) 样本拒绝分布

- total_rows：9787
- learnable_rows：9608
- rejected_rows：179

| reason | count |
| --- | --- |
| pending_next_snapshot | 179 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：9608
- pos/neg：1607/8001
- feature_coverage：1.0
- pass_trade_dates：131
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
