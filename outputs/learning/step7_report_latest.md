# Step7 自学习报告（latest）

- 生成时间：2026-07-14 20:27:03
- RunMode：auto_daily
- Today：20260714
- LatestSnapshot：20260714
- LabelUpperBound：20260714

## 1) 最新命中

- trade_date：20260713
- verify_date：20260714
- hit/topn：4/10
- hit_rate：0.4
- top1：1/1，hit_rate=1.0
- top3：2/3，hit_rate=0.6667
- top5：2/5，hit_rate=0.4
- top10：4/10，hit_rate=0.4
- note：src=feature_history_v3;ranking=published_file:pred_top10_20260713.csv

## 1.1) 近10日发布排名命中率（done-only）

| trade_date | verify_date | top1_hit_rate | top3_hit_rate | top5_hit_rate | top10_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 20260630 | 20260701 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260701 | 20260702 | 0.0 | 0.0 | 0.2 | 0.3 |
| 20260702 | 20260703 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260703 | 20260706 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260706 | 20260707 | 0.0 | 0.0 | 0.0 | 0.1 |
| 20260707 | 20260708 | 0.0 | 0.0 | 0.0 | 0.2 |
| 20260708 | 20260709 | 0.0 | 0.0 | 0.2 | 0.1 |
| 20260709 | 20260710 | 0.0 | 0.3333 | 0.2 | 0.1 |
| 20260710 | 20260713 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20260713 | 20260714 | 1.0 | 0.6667 | 0.4 | 0.4 |

## 1.2) 发布排名累计指标

| rank | trade_days | sample_count | hit_count | hit_rate |
| --- | --- | --- | --- | --- |
| Top1 | 110 | 110 | 60 | 0.5455 |
| Top3 | 110 | 330 | 151 | 0.4576 |
| Top5 | 110 | 550 | 221 | 0.4018 |
| Top10 | 110 | 1100 | 366 | 0.3327 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：112
- pass_dates：110
- fail_dates：2
- eligible_train_rows：8036

## 2.1) 样本拒绝分布

- total_rows：8211
- learnable_rows：8036
- rejected_rows：175

| reason | count |
| --- | --- |
| pending_next_snapshot | 175 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：8036
- pos/neg：1322/6714
- feature_coverage：1.0
- pass_trade_dates：110
- fail_trade_dates：2
- reason：ok_partial_pass_dates_model_updated

## 4) Warnings

- next_trade_snapshot_missing: trade_date=20260407, expected_verify_date=20260408
