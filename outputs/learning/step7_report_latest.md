# Step7 自学习报告（latest）

- 生成时间：2026-03-15 20:23:08
- RunMode：auto_daily
- Today：20260315
- LatestSnapshot：20260313
- LabelUpperBound：20260313

## 1) 最新命中

- trade_date：20260312
- verify_date：20260313
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260227 | 20260302 | 10 | 2 | 0.2 |
| 20260302 | 20260303 | 10 | 7 | 0.7 |
| 20260303 | 20260304 | 10 | 6 | 0.6 |
| 20260304 | 20260305 | 10 | 2 | 0.2 |
| 20260305 | 20260306 | 10 | 4 | 0.4 |
| 20260306 | 20260309 | 10 | 5 | 0.5 |
| 20260309 | 20260310 | 10 | 5 | 0.5 |
| 20260310 | 20260311 | 10 | 10 | 1.0 |
| 20260311 | 20260312 | 10 | 5 | 0.5 |
| 20260312 | 20260313 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：31
- pass_dates：30
- fail_dates：1
- eligible_train_rows：2087

## 2.1) 样本拒绝分布

- total_rows：2146
- learnable_rows：2087
- rejected_rows：59

| reason | count |
| --- | --- |
| pending_next_snapshot | 59 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：2087
- pos/neg：370/1717
- feature_coverage：1.0
- pass_trade_dates：30
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
