# Step7 自学习报告（latest）

- 生成时间：2026-06-18 15:51:52
- RunMode：auto_daily
- Today：20260618
- LatestSnapshot：20260617
- LabelUpperBound：20260617

## 1) 最新命中

- trade_date：20260616
- verify_date：20260617
- hit/topn：4/10
- hit_rate：0.4
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260603 | 20260604 | 10 | 3 | 0.3 |
| 20260604 | 20260605 | 10 | 1 | 0.1 |
| 20260605 | 20260608 | 10 | 1 | 0.1 |
| 20260608 | 20260609 | 10 | 1 | 0.1 |
| 20260609 | 20260610 | 10 | 4 | 0.4 |
| 20260610 | 20260611 | 10 | 3 | 0.3 |
| 20260611 | 20260612 | 10 | 4 | 0.4 |
| 20260612 | 20260615 | 10 | 2 | 0.2 |
| 20260615 | 20260616 | 10 | 2 | 0.2 |
| 20260616 | 20260617 | 10 | 4 | 0.4 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：94
- pass_dates：93
- fail_dates：1
- eligible_train_rows：6528

## 2.1) 样本拒绝分布

- total_rows：6616
- learnable_rows：6528
- rejected_rows：88

| reason | count |
| --- | --- |
| pending_next_snapshot | 88 |

## 3) 训练执行结果

- trained：False
- updated：False
- level：level3
- train_rows：6528
- pos/neg：1097/5431
- feature_coverage：1.0
- pass_trade_dates：93
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
