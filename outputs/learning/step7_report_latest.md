# Step7 自学习报告（latest）

- 生成时间：2026-04-27 21:07:50
- RunMode：auto_daily
- Today：20260427
- LatestSnapshot：20260427
- LabelUpperBound：20260427

## 1) 最新命中

- trade_date：20260424
- verify_date：20260427
- hit/topn：2/10
- hit_rate：0.2
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260413 | 20260414 | 10 | 3 | 0.3 |
| 20260414 | 20260415 | 10 | 4 | 0.4 |
| 20260415 | 20260416 | 10 | 3 | 0.3 |
| 20260416 | 20260417 | 10 | 3 | 0.3 |
| 20260417 | 20260420 | 10 | 2 | 0.2 |
| 20260420 | 20260421 | 10 | 3 | 0.3 |
| 20260421 | 20260422 | 10 | 3 | 0.3 |
| 20260422 | 20260423 | 10 | 4 | 0.4 |
| 20260423 | 20260424 | 10 | 5 | 0.5 |
| 20260424 | 20260427 | 10 | 2 | 0.2 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：60
- pass_dates：59
- fail_dates：1
- eligible_train_rows：3717

## 2.1) 样本拒绝分布

- total_rows：3782
- learnable_rows：3717
- rejected_rows：65

| reason | count |
| --- | --- |
| pending_next_snapshot | 65 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：3717
- pos/neg：632/3085
- feature_coverage：1.0
- pass_trade_dates：59
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
