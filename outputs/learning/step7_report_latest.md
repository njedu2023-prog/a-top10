# Step7 自学习报告（latest）

- 生成时间：2026-03-15 16:19:22
- RunMode：auto_daily
- Today：20260315
- LatestSnapshot：20260313
- LabelUpperBound：20260313
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260312
- expected_next_trade_date：20260313
- actual_next_trade_date：20260313
- hit/topn：0/10
- hit_rate：0.0
- note：src=pred_top10_history

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | actual_next_trade_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260227 | 20260302 | 10 | 1 | 0.1 |
| 20260302 | 20260303 | 10 | 7 | 0.7 |
| 20260303 | 20260304 | 10 | 2 | 0.2 |
| 20260304 | 20260305 | 10 | 3 | 0.3 |
| 20260305 | 20260306 | 10 | 3 | 0.3 |
| 20260306 | 20260309 | 10 | 3 | 0.3 |
| 20260309 | 20260310 | 10 | 0 | 0.0 |
| 20260310 | 20260311 | 10 | 2 | 0.2 |
| 20260311 | 20260312 | 10 | 1 | 0.1 |
| 20260312 | 20260313 | 10 | 0 | 0.0 |

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：2146
- 过滤后行数：2058
- 丢弃全零特征行：29
- 日期总数：31
- 使用日期：31
- 特征覆盖率：0.7289309013446944

## 3) 训练执行结果

- trained：False
- updated：False
- lr_saved：False
- lgbm_saved：False
- models_dir：
- level：level3
- train_rows：2058
- pos/neg：368/1690
- feature_coverage：0.7289309013446944
- reason：skip_train: quality_gate_fail

## 4) Warnings

- label_pending: trade_date=20260313 no next snapshot
- skip_train: quality_gate_fail
