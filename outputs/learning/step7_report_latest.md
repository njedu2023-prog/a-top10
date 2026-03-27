# Step7 自学习报告（latest）

- 生成时间：2026-03-27 20:08:07
- RunMode：auto_daily
- Today：20260327
- LatestSnapshot：20260327
- LabelUpperBound：20260327

## 1) 最新命中

- trade_date：20260326
- verify_date：20260327
- hit/topn：4/10
- hit_rate：0.4
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260313 | 20260316 | 10 | 2 | 0.2 |
| 20260316 | 20260317 | 10 | 6 | 0.6 |
| 20260317 | 20260318 | 10 | 2 | 0.2 |
| 20260318 | 20260319 | 10 | 4 | 0.4 |
| 20260319 | 20260320 | 10 | 5 | 0.5 |
| 20260320 | 20260323 | 10 | 2 | 0.2 |
| 20260323 | 20260324 | 10 | 4 | 0.4 |
| 20260324 | 20260325 | 10 | 4 | 0.4 |
| 20260325 | 20260326 | 10 | 3 | 0.3 |
| 20260326 | 20260327 | 10 | 4 | 0.4 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：41
- pass_dates：40
- fail_dates：1
- eligible_train_rows：2581

## 2.1) 样本拒绝分布

- total_rows：2660
- learnable_rows：2581
- rejected_rows：79

| reason | count |
| --- | --- |
| pending_next_snapshot | 79 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：2581
- pos/neg：464/2117
- feature_coverage：1.0
- pass_trade_dates：40
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
