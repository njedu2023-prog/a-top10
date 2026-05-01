# Step7 自学习报告（latest）

- 生成时间：2026-05-02 00:13:08
- RunMode：auto_daily
- Today：20260502
- LatestSnapshot：20260430
- LabelUpperBound：20260430

## 1) 最新命中

- trade_date：20260429
- verify_date：20260430
- hit/topn：5/10
- hit_rate：0.5
- note：src=feature_history_v3

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | verify_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260416 | 20260417 | 10 | 3 | 0.3 |
| 20260417 | 20260420 | 10 | 2 | 0.2 |
| 20260420 | 20260421 | 10 | 3 | 0.3 |
| 20260421 | 20260422 | 10 | 3 | 0.3 |
| 20260422 | 20260423 | 10 | 4 | 0.4 |
| 20260423 | 20260424 | 10 | 5 | 0.5 |
| 20260424 | 20260427 | 10 | 2 | 0.2 |
| 20260427 | 20260428 | 10 | 2 | 0.2 |
| 20260428 | 20260429 | 10 | 2 | 0.2 |
| 20260429 | 20260430 | 10 | 5 | 0.5 |

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：63
- pass_dates：62
- fail_dates：1
- eligible_train_rows：3943

## 2.1) 样本拒绝分布

- total_rows：4022
- learnable_rows：3943
- rejected_rows：79

| reason | count |
| --- | --- |
| pending_next_snapshot | 79 |

## 3) 训练执行结果

- trained：True
- updated：True
- level：level3
- train_rows：3943
- pos/neg：672/3271
- feature_coverage：1.0
- pass_trade_dates：62
- fail_trade_dates：1
- reason：ok_partial_pass_dates_trained
