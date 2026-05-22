# Step7 自学习报告（latest）

- 生成时间：2026-05-22 21:57:18
- RunMode：auto_daily
- Today：20260522
- LatestSnapshot：20260522
- LabelUpperBound：20260522

## 1) 最新命中

- trade_date：20260521
- verify_date：20260522
- hit/topn：5/10
- hit_rate：0.5
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260508 | 20260511 | 10 | 2 | 0.2 |
| 20260511 | 20260512 | 10 | 4 | 0.4 |
| 20260512 | 20260513 | 10 | 4 | 0.4 |
| 20260513 | 20260514 | 10 | 3 | 0.3 |
| 20260514 | 20260515 | 10 | 3 | 0.3 |
| 20260515 | 20260518 | 10 | 3 | 0.3 |
| 20260518 | 20260519 | 10 | 6 | 0.6 |
| 20260519 | 20260520 | 10 | 3 | 0.3 |
| 20260520 | 20260521 | 10 | 3 | 0.3 |
| 20260521 | 20260522 | 10 | 5 | 0.5 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：76
- pass_dates：75
- fail_dates：1
- eligible_train_rows：4967

## 2.1) 样本拒绝分布

- total_rows：5083
- learnable_rows：4967
- rejected_rows：116

| reason | count |
| --- | --- |
| pending_next_snapshot | 116 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：4967
- pos/neg：859/4108
- feature_coverage：1.0
- pass_trade_dates：75
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
