# Step7 自学习报告（latest）

- 生成时间：2026-06-26 20:51:02
- RunMode：auto_daily
- Today：20260626
- LatestSnapshot：20260626
- LabelUpperBound：20260626

## 2) 批级闸门

- pass：True
- reason：partial_pass_bad_trade_dates_excluded
- trade_dates：100
- pass_dates：99
- fail_dates：1
- eligible_train_rows：7129

## 2.1) 样本拒绝分布

- total_rows：7189
- learnable_rows：7129
- rejected_rows：60

| reason | count |
| --- | --- |
| pending_next_snapshot | 60 |

## 3) 训练执行结果

- trained：False
- updated：False
- level：level3
- train_rows：7129
- pos/neg：1193/5936
- feature_coverage：1.0
- pass_trade_dates：99
- fail_dates：1
- reason：ok_partial_pass_dates_trained
