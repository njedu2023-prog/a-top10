# Step7 自学习报告（latest）

- 生成时间：2026-04-29 21:10:27
- RunMode：auto_daily
- Today：20260429
- LatestSnapshot：20260429
- LabelUpperBound：20260429

## 1) 最新命中

- trade_date：20260428
- verify_date：20260429
- hit/topn：2/10
- hit_rate：0.2
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260415 | 20260416 | 10 | 3 | 0.3 |
| 20260416 | 20260417 | 10 | 3 | 0.3 |
| 20260417 | 20260420 | 10 | 2 | 0.2 |
| 20260420 | 20260421 | 10 | 3 | 0.3 |
| 20260421 | 20260422 | 10 | 3 | 0.3 |
| 20260422 | 20260423 | 10 | 4 | 0.4 |
| 20260423 | 20260424 | 10 | 5 | 0.5 |
| 20260424 | 20260427 | 10 | 2 | 0.2 |
| 20260427 | 20260428 | 10 | 2 | 0.2 |
| 20260428 | 20260429 | 10 | 2 | 0.2 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：62
- pass_dates：61
- fail_dates：1
- eligible_train_rows：3841

## 2.1) 样本拒绝分布

- total_rows：3943
- learnable_rows：3841
- rejected_rows：102

| reason | count |
| --- | --- |
| pending_next_snapshot | 102 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：3841
- pos/neg：654/3187
- feature_coverage：1.0
- pass_trade_dates：61
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
