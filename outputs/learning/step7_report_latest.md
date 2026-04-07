# Step7 自学习报告（latest）

- 生成时间：2026-04-07 20:40:16
- RunMode：auto_daily
- Today：20260407
- LatestSnapshot：20260407
- LabelUpperBound：20260407

## 1) 最新命中

- trade_date：20260403
- verify_date：20260407
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260323 | 20260324 | 10 | 4 | 0.4 |
| 20260324 | 20260325 | 10 | 4 | 0.4 |
| 20260325 | 20260326 | 10 | 3 | 0.3 |
| 20260326 | 20260327 | 10 | 4 | 0.4 |
| 20260327 | 20260330 | 10 | 3 | 0.3 |
| 20260330 | 20260331 | 10 | 0 | 0.0 |
| 20260331 | 20260401 | 10 | 0 | 0.0 |
| 20260401 | 20260402 | 10 | 2 | 0.2 |
| 20260402 | 20260403 | 10 | 3 | 0.3 |
| 20260403 | 20260407 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：47
- pass_dates：46
- fail_dates：1
- eligible_train_rows：2897

## 2.1) 样本拒绝分布

- total_rows：2990
- learnable_rows：2897
- rejected_rows：93

| reason | count |
| --- | --- |
| pending_next_snapshot | 93 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：2897
- pos/neg：508/2389
- feature_coverage：1.0
- pass_trade_dates：46
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
