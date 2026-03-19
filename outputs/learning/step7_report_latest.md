# Step7 自学习报告（latest）

- 生成时间：2026-03-19 20:01:55
- RunMode：auto_daily
- Today：20260319
- LatestSnapshot：20260319
- LabelUpperBound：20260319

## 1) 最新命中

- trade_date：20260318
- verify_date：20260319
- hit/topn：4/10
- hit_rate：0.4
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260305 | 20260306 | 10 | 8 | 0.8 |
| 20260306 | 20260309 | 10 | 10 | 1.0 |
| 20260309 | 20260310 | 10 | 6 | 0.6 |
| 20260310 | 20260311 | 10 | 10 | 1.0 |
| 20260311 | 20260312 | 10 | 8 | 0.8 |
| 20260312 | 20260313 | 10 | 8 | 0.8 |
| 20260313 | 20260316 | 10 | 2 | 0.2 |
| 20260316 | 20260317 | 10 | 6 | 0.6 |
| 20260317 | 20260318 | 10 | 2 | 0.2 |
| 20260318 | 20260319 | 10 | 4 | 0.4 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：35
- pass_dates：34
- fail_dates：1
- eligible_train_rows：2289

## 2.1) 样本拒绝分布

- total_rows：2317
- learnable_rows：2289
- rejected_rows：28

| reason | count |
| --- | --- |
| pending_next_snapshot | 28 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：2289
- pos/neg：402/1887
- feature_coverage：1.0
- pass_trade_dates：34
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
