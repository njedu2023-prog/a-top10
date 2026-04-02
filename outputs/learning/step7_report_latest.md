# Step7 自学习报告（latest）

- 生成时间：2026-04-02 20:36:11
- RunMode：auto_daily
- Today：20260402
- LatestSnapshot：20260402
- LabelUpperBound：20260402

## 1) 最新命中

- trade_date：20260401
- verify_date：20260402
- hit/topn：2/10
- hit_rate：0.2
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260319 | 20260320 | 10 | 5 | 0.5 |
| 20260320 | 20260323 | 10 | 2 | 0.2 |
| 20260323 | 20260324 | 10 | 4 | 0.4 |
| 20260324 | 20260325 | 10 | 4 | 0.4 |
| 20260325 | 20260326 | 10 | 3 | 0.3 |
| 20260326 | 20260327 | 10 | 4 | 0.4 |
| 20260327 | 20260330 | 10 | 3 | 0.3 |
| 20260330 | 20260331 | 10 | 0 | 0.0 |
| 20260331 | 20260401 | 10 | 0 | 0.0 |
| 20260401 | 20260402 | 10 | 2 | 0.2 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：45
- pass_dates：44
- fail_dates：1
- eligible_train_rows：2833

## 2.1) 样本拒绝分布

- total_rows：2861
- learnable_rows：2833
- rejected_rows：28

| reason | count |
| --- | --- |
| pending_next_snapshot | 28 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：2833
- pos/neg：497/2336
- feature_coverage：1.0
- pass_trade_dates：44
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
