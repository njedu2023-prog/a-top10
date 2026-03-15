# Step7 自学习报告（latest）

- 生成时间：2026-03-15 17:30:30
- RunMode：auto_daily
- Today：20260315
- LatestSnapshot：20260313
- LabelUpperBound：20260313

## 1) 最新命中

- trade_date：20260312
- verify_date：20260313
- hit/topn：8/10
- hit_rate：0.8
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260227 | 20260302 | 10 | 1 | 0.1 |
| 20260302 | 20260303 | 10 | 7 | 0.7 |
| 20260303 | 20260304 | 10 | 2 | 0.2 |
| 20260304 | 20260305 | 10 | 3 | 0.3 |
| 20260305 | 20260306 | 10 | 3 | 0.3 |
| 20260306 | 20260309 | 10 | 3 | 0.3 |
| 20260309 | 20260310 | 10 | 0 | 0.0 |
| 20260310 | 20260311 | 10 | 10 | 1.0 |
| 20260311 | 20260312 | 10 | 8 | 0.8 |
| 20260312 | 20260313 | 10 | 8 | 0.8 |

## 2) 批级闸门

- pass：False
- reason：one_or_more_trade_dates_failed_batch_gate
- trade_dates：31

## 3) 训练执行结果

- trained：False
- updated：False
- level：level3
- train_rows：1926
- pos/neg：344/1582
- feature_coverage：1.0
- reason：skip_train: batch_gate_fail

## 4) Warnings

- skip_train: batch_gate_fail
