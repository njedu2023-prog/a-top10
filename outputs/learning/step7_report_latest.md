# Step7 自学习报告（latest）

- 生成时间：2026-02-27 06:37:27
- Today：20260227
- LatestSnapshot：20260226
- LabelUpperBound：20260226
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260225
- expected_next_trade_date：20260226
- actual_next_trade_date：20260226
- hit/topn：4/10
- hit_rate：0.4
- note：src=pred_top10_history

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | actual_next_trade_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260204 | 20260205 | 10 | 5 | 0.5 |
| 20260205 | 20260206 | 10 | 4 | 0.4 |
| 20260206 | 20260209 | 10 | 6 | 0.6 |
| 20260209 | 20260210 | 10 | 6 | 0.6 |
| 20260210 | 20260211 | 10 | 4 | 0.4 |
| 20260211 | 20260212 | 10 | 7 | 0.7 |
| 20260212 | 20260213 | 10 | 3 | 0.3 |
| 20260213 | 20260224 | 10 | 3 | 0.3 |
| 20260224 | 20260225 | 10 | 5 | 0.5 |
| 20260225 | 20260226 | 10 | 4 | 0.4 |

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：1451
- 过滤后行数：1359
- 丢弃全零特征行：29
- 日期总数：20
- 使用日期：20

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：1359
- pos/neg：236/1123
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
