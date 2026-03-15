# Step7 自学习报告（latest）

- 生成时间：2026-03-15 18:17:23
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
| 20260306 | 20260309 | 10 | 5 | 0.5 |
| 20260309 | 20260310 | 10 | 5 | 0.5 |
| 20260310 | 20260311 | 10 | 5 | 0.5 |
| 20260311 | 20260312 | 10 | 8 | 0.8 |
| 20260312 | 20260313 | 10 | 8 | 0.8 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：31
- pass_dates：28
- fail_dates：3
- eligible_train_rows：1982

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：1982
- pos/neg：354/1628
- feature_coverage：1.0
- pass_trade_dates：28
- fail_trade_dates：3
- reason：ok_partial_pass_dates_trained
