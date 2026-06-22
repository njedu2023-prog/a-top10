# Step7 自学习报告（latest）

- 生成时间：2026-06-22 21:27:53
- RunMode：auto_daily
- Today：20260622
- LatestSnapshot：20260622
- LabelUpperBound：20260622

## 1) 最新命中

- trade_date：20260618
- verify_date：20260622
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260605 | 20260608 | 10 | 1 | 0.1 |
| 20260608 | 20260609 | 10 | 1 | 0.1 |
| 20260609 | 20260610 | 10 | 4 | 0.4 |
| 20260610 | 20260611 | 10 | 3 | 0.3 |
| 20260611 | 20260612 | 10 | 4 | 0.4 |
| 20260612 | 20260615 | 10 | 2 | 0.2 |
| 20260615 | 20260616 | 10 | 2 | 0.2 |
| 20260616 | 20260617 | 10 | 4 | 0.4 |
| 20260617 | 20260618 | 10 | 1 | 0.1 |
| 20260618 | 20260622 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：96
- pass_dates：95
- fail_dates：1
- eligible_train_rows：6709

## 2.1) 样本拒绝分布

- total_rows：6847
- learnable_rows：6709
- rejected_rows：138

| reason | count |
| --- | --- |
| pending_next_snapshot | 138 |

## 3) 训练执行结果

- trained：False
- updated：False
- level：level3
- train_rows：6709
- pos/neg：1135/5574
- feature_coverage：1.0
- pass_trade_dates：95
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
