# Step7 自学习报告（latest）

- 生成时间：2026-04-10 20:31:22
- RunMode：auto_daily
- Today：20260410
- LatestSnapshot：20260410
- LabelUpperBound：20260410

## 1) 最新命中

- trade_date：20260409
- verify_date：20260410
- hit/topn：5/10
- hit_rate：0.5
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260325 | 20260326 | 10 | 3 | 0.3 |
| 20260326 | 20260327 | 10 | 4 | 0.4 |
| 20260327 | 20260330 | 10 | 3 | 0.3 |
| 20260330 | 20260331 | 10 | 0 | 0.0 |
| 20260331 | 20260401 | 10 | 0 | 0.0 |
| 20260401 | 20260402 | 10 | 2 | 0.2 |
| 20260402 | 20260403 | 10 | 3 | 0.3 |
| 20260403 | 20260407 | 10 | 3 | 0.3 |
| 20260407 | 20260409 | 10 | 1 | 0.1 |
| 20260409 | 20260410 | 10 | 5 | 0.5 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：49
- pass_dates：48
- fail_dates：1
- eligible_train_rows：3044

## 2.1) 样本拒绝分布

- total_rows：3103
- learnable_rows：3044
- rejected_rows：59

| reason | count |
| --- | --- |
| pending_next_snapshot | 59 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：3044
- pos/neg：525/2519
- feature_coverage：1.0
- pass_trade_dates：48
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
