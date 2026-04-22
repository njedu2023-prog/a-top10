# Step7 自学习报告（latest）

- 生成时间：2026-04-22 20:43:17
- RunMode：auto_daily
- Today：20260422
- LatestSnapshot：20260422
- LabelUpperBound：20260422

## 1) 最新命中

- trade_date：20260421
- verify_date：20260422
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260407 | 20260409 | 10 | 1 | 0.1 |
| 20260409 | 20260410 | 10 | 5 | 0.5 |
| 20260410 | 20260413 | 10 | 2 | 0.2 |
| 20260413 | 20260414 | 10 | 3 | 0.3 |
| 20260414 | 20260415 | 10 | 4 | 0.4 |
| 20260415 | 20260416 | 10 | 3 | 0.3 |
| 20260416 | 20260417 | 10 | 3 | 0.3 |
| 20260417 | 20260420 | 10 | 2 | 0.2 |
| 20260420 | 20260421 | 10 | 3 | 0.3 |
| 20260421 | 20260422 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：57
- pass_dates：56
- fail_dates：1
- eligible_train_rows：3553

## 2.1) 样本拒绝分布

- total_rows：3608
- learnable_rows：3553
- rejected_rows：55

| reason | count |
| --- | --- |
| pending_next_snapshot | 55 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：3553
- pos/neg：606/2947
- feature_coverage：1.0
- pass_trade_dates：56
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
