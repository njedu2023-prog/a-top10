# Step7 自学习报告（latest）

- 生成时间：2026-05-27 00:53:52
- RunMode：auto_daily
- Today：20260527
- LatestSnapshot：20260526
- LabelUpperBound：20260526

## 1) 最新命中

- trade_date：20260525
- verify_date：20260526
- hit/topn：0/10
- hit_rate：0.0
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260512 | 20260513 | 10 | 4 | 0.4 |
| 20260513 | 20260514 | 10 | 3 | 0.3 |
| 20260514 | 20260515 | 10 | 3 | 0.3 |
| 20260515 | 20260518 | 10 | 3 | 0.3 |
| 20260518 | 20260519 | 10 | 6 | 0.6 |
| 20260519 | 20260520 | 10 | 3 | 0.3 |
| 20260520 | 20260521 | 10 | 3 | 0.3 |
| 20260521 | 20260522 | 10 | 5 | 0.5 |
| 20260522 | 20260525 | 10 | 4 | 0.4 |
| 20260525 | 20260526 | 10 | 0 | 0.0 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：78
- pass_dates：77
- fail_dates：1
- eligible_train_rows：5186

## 2.1) 样本拒绝分布

- total_rows：5235
- learnable_rows：5186
- rejected_rows：49

| reason | count |
| --- | --- |
| pending_next_snapshot | 49 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：5186
- pos/neg：895/4291
- feature_coverage：1.0
- pass_trade_dates：77
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
