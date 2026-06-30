# Step7 自学习报告（latest）

- 生成时间：2026-06-30 21:26:00
- RunMode：auto_daily
- Today：20260630
- LatestSnapshot：20260630
- LabelUpperBound：20260630

## 1) 最新命中

- trade_date：20260629
- verify_date：20260630
- hit/topn：2/10
- hit_rate：0.2
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260615 | 20260616 | 10 | 2 | 0.2 |
| 20260616 | 20260617 | 10 | 4 | 0.4 |
| 20260617 | 20260618 | 10 | 1 | 0.1 |
| 20260618 | 20260622 | 10 | 3 | 0.3 |
| 20260622 | 20260623 | 10 | 3 | 0.3 |
| 20260623 | 20260624 | 10 | 4 | 0.4 |
| 20260624 | 20260625 | 10 | 2 | 0.2 |
| 20260625 | 20260626 | 10 | 3 | 0.3 |
| 20260626 | 20260629 | 10 | 3 | 0.3 |
| 20260629 | 20260630 | 10 | 2 | 0.2 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：102
- pass_dates：101
- fail_dates：1
- eligible_train_rows：7295

## 2.1) 样本拒绝分布

- total_rows：7434
- learnable_rows：7295
- rejected_rows：139

| reason | count |
| --- | --- |
| pending_next_snapshot | 139 |

## 3) 训练执行结果

- trained：False
- updated：False
- level：level3
- train_rows：7295
- pos/neg：1221/6074
- feature_coverage：1.0
- pass_trade_dates：101
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
