# Step7 自学习报告（latest）

- 生成时间：2026-06-10 22:38:42
- RunMode：auto_daily
- Today：20260610
- LatestSnapshot：20260610
- LabelUpperBound：20260610

## 1) 最新命中

- trade_date：20260609
- verify_date：20260610
- hit/topn：4/10
- hit_rate：0.4
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260527 | 20260528 | 10 | 3 | 0.3 |
| 20260528 | 20260529 | 10 | 2 | 0.2 |
| 20260529 | 20260601 | 10 | 1 | 0.1 |
| 20260601 | 20260602 | 10 | 4 | 0.4 |
| 20260602 | 20260603 | 10 | 1 | 0.1 |
| 20260603 | 20260604 | 10 | 3 | 0.3 |
| 20260604 | 20260605 | 10 | 1 | 0.1 |
| 20260605 | 20260608 | 10 | 1 | 0.1 |
| 20260608 | 20260609 | 10 | 1 | 0.1 |
| 20260609 | 20260610 | 10 | 4 | 0.4 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：89
- pass_dates：88
- fail_dates：1
- eligible_train_rows：6030

## 2.1) 样本拒绝分布

- total_rows：6101
- learnable_rows：6030
- rejected_rows：71

| reason | count |
| --- | --- |
| pending_next_snapshot | 71 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：6030
- pos/neg：1015/5015
- feature_coverage：1.0
- pass_trade_dates：88
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
