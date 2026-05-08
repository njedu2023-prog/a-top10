# Step7 自学习报告（latest）

- 生成时间：2026-05-08 20:56:42
- RunMode：auto_daily
- Today：20260508
- LatestSnapshot：20260508
- LabelUpperBound：20260508

## 1) 最新命中

- trade_date：20260507
- verify_date：20260508
- hit/topn：1/10
- hit_rate：0.1
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260421 | 20260422 | 10 | 3 | 0.3 |
| 20260422 | 20260423 | 10 | 4 | 0.4 |
| 20260423 | 20260424 | 10 | 5 | 0.5 |
| 20260424 | 20260427 | 10 | 2 | 0.2 |
| 20260427 | 20260428 | 10 | 2 | 0.2 |
| 20260428 | 20260429 | 10 | 2 | 0.2 |
| 20260429 | 20260430 | 10 | 5 | 0.5 |
| 20260430 | 20260506 | 10 | 3 | 0.3 |
| 20260506 | 20260507 | 10 | 3 | 0.3 |
| 20260507 | 20260508 | 10 | 1 | 0.1 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：66
- pass_dates：65
- fail_dates：1
- eligible_train_rows：4225

## 2.1) 样本拒绝分布

- total_rows：4323
- learnable_rows：4225
- rejected_rows：98

| reason | count |
| --- | --- |
| pending_next_snapshot | 98 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：4225
- pos/neg：727/3498
- feature_coverage：1.0
- pass_trade_dates：65
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
