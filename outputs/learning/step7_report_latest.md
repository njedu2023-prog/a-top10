# Step7 自学习报告（latest）

- 生成时间：2026-07-06 22:44:50
- RunMode：auto_daily
- Today：20260706
- LatestSnapshot：20260706
- LabelUpperBound：20260706

## 1) 最新命中

- trade_date：20260703
- verify_date：20260706
- hit/topn：1/10
- hit_rate：0.1
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260622 | 20260623 | 10 | 3 | 0.3 |
| 20260623 | 20260624 | 10 | 4 | 0.4 |
| 20260624 | 20260625 | 10 | 2 | 0.2 |
| 20260625 | 20260626 | 10 | 3 | 0.3 |
| 20260626 | 20260629 | 10 | 3 | 0.3 |
| 20260629 | 20260630 | 10 | 2 | 0.2 |
| 20260630 | 20260701 | 10 | 1 | 0.1 |
| 20260701 | 20260702 | 10 | 3 | 0.3 |
| 20260702 | 20260703 | 10 | 0 | 0.0 |
| 20260703 | 20260706 | 10 | 1 | 0.1 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：106
- pass_dates：105
- fail_dates：1
- eligible_train_rows：7789

## 2.1) 样本拒绝分布

- total_rows：7855
- learnable_rows：7789
- rejected_rows：66

| reason | count |
| --- | --- |
| pending_next_snapshot | 66 |

## 3) 训练执行结果

- trained：False
- updated：False
- level：level3
- train_rows：7789
- pos/neg：1285/6504
- feature_coverage：1.0
- pass_trade_dates：105
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
