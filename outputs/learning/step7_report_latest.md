# Step7 自学习报告（latest）

- 生成时间：2026-06-01 21:15:12
- RunMode：auto_daily
- Today：20260601
- LatestSnapshot：20260601
- LabelUpperBound：20260601

## 1) 最新命中

- trade_date：20260529
- verify_date：20260601
- hit/topn：1/10
- hit_rate：0.1
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260518 | 20260519 | 10 | 6 | 0.6 |
| 20260519 | 20260520 | 10 | 3 | 0.3 |
| 20260520 | 20260521 | 10 | 3 | 0.3 |
| 20260521 | 20260522 | 10 | 5 | 0.5 |
| 20260522 | 20260525 | 10 | 4 | 0.4 |
| 20260525 | 20260526 | 10 | 0 | 0.0 |
| 20260526 | 20260527 | 10 | 4 | 0.4 |
| 20260527 | 20260528 | 10 | 3 | 0.3 |
| 20260528 | 20260529 | 10 | 2 | 0.2 |
| 20260529 | 20260601 | 10 | 1 | 0.1 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：82
- pass_dates：81
- fail_dates：1
- eligible_train_rows：5435

## 2.1) 样本拒绝分布

- total_rows：5555
- learnable_rows：5435
- rejected_rows：120

| reason | count |
| --- | --- |
| pending_next_snapshot | 120 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：5435
- pos/neg：932/4503
- feature_coverage：1.0
- pass_trade_dates：81
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
