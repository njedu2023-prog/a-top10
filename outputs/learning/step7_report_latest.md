# Step7 自学习报告（latest）

- 生成时间：2026-05-15 21:16:20
- RunMode：auto_daily
- Today：20260515
- LatestSnapshot：20260515
- LabelUpperBound：20260515

## 1) 最新命中

- trade_date：20260514
- verify_date：20260515
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260428 | 20260429 | 10 | 2 | 0.2 |
| 20260429 | 20260430 | 10 | 5 | 0.5 |
| 20260430 | 20260506 | 10 | 3 | 0.3 |
| 20260506 | 20260507 | 10 | 3 | 0.3 |
| 20260507 | 20260508 | 10 | 1 | 0.1 |
| 20260508 | 20260511 | 10 | 2 | 0.2 |
| 20260511 | 20260512 | 10 | 4 | 0.4 |
| 20260512 | 20260513 | 10 | 4 | 0.4 |
| 20260513 | 20260514 | 10 | 3 | 0.3 |
| 20260514 | 20260515 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：71
- pass_dates：70
- fail_dates：1
- eligible_train_rows：4644

## 2.1) 样本拒绝分布

- total_rows：4699
- learnable_rows：4644
- rejected_rows：55

| reason | count |
| --- | --- |
| pending_next_snapshot | 55 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：4644
- pos/neg：803/3841
- feature_coverage：1.0
- pass_trade_dates：70
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
