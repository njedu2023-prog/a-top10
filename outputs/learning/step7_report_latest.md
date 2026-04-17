# Step7 自学习报告（latest）

- 生成时间：2026-04-17 20:40:54
- RunMode：auto_daily
- Today：20260417
- LatestSnapshot：20260417
- LabelUpperBound：20260417

## 1) 最新命中

- trade_date：20260416
- verify_date：20260417
- hit/topn：3/10
- hit_rate：0.3
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260401 | 20260402 | 10 | 2 | 0.2 |
| 20260402 | 20260403 | 10 | 3 | 0.3 |
| 20260403 | 20260407 | 10 | 3 | 0.3 |
| 20260407 | 20260409 | 10 | 1 | 0.1 |
| 20260409 | 20260410 | 10 | 5 | 0.5 |
| 20260410 | 20260413 | 10 | 2 | 0.2 |
| 20260413 | 20260414 | 10 | 3 | 0.3 |
| 20260414 | 20260415 | 10 | 4 | 0.4 |
| 20260415 | 20260416 | 10 | 3 | 0.3 |
| 20260416 | 20260417 | 10 | 3 | 0.3 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：54
- pass_dates：53
- fail_dates：1
- eligible_train_rows：3353

## 2.1) 样本拒绝分布

- total_rows：3424
- learnable_rows：3353
- rejected_rows：71

| reason | count |
| --- | --- |
| pending_next_snapshot | 71 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：3353
- pos/neg：575/2778
- feature_coverage：1.0
- pass_trade_dates：53
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
