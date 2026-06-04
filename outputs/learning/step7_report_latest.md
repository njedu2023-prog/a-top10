# Step7 自学习报告（latest）

- 生成时间：2026-06-04 22:15:27
- RunMode：auto_daily
- Today：20260604
- LatestSnapshot：20260604
- LabelUpperBound：20260604

## 1) 最新命中

- trade_date：20260603
- verify_date：20260604
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260521 | 20260522 | 10 | 5 | 0.5 |
| 20260522 | 20260525 | 10 | 4 | 0.4 |
| 20260525 | 20260526 | 10 | 0 | 0.0 |
| 20260526 | 20260527 | 10 | 4 | 0.4 |
| 20260527 | 20260528 | 10 | 3 | 0.3 |
| 20260528 | 20260529 | 10 | 2 | 0.2 |
| 20260529 | 20260601 | 10 | 1 | 0.1 |
| 20260601 | 20260602 | 10 | 4 | 0.4 |
| 20260602 | 20260603 | 10 | 1 | 0.1 |
| 20260603 | 20260604 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：85
- pass_dates：84
- fail_dates：1
- eligible_train_rows：5687

## 2.1) 样本拒绝分布

- total_rows：5767
- learnable_rows：5687
- rejected_rows：80

| reason | count |
| --- | --- |
| pending_next_snapshot | 80 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：5687
- pos/neg：969/4718
- feature_coverage：1.0
- pass_trade_dates：84
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
