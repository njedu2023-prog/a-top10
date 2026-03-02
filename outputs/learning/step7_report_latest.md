# Step7 自学习报告（latest）

- 生成时间：2026-03-02 19:57:24
- Today：20260302
- LatestSnapshot：20260302
- LabelUpperBound：20260302
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260227
- expected_next_trade_date：20260302
- actual_next_trade_date：20260302
- hit/topn：1/10
- hit_rate：0.1
- note：src=pred_top10_history

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | actual_next_trade_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260206 | 20260209 | 10 | 6 | 0.6 |
| 20260209 | 20260210 | 10 | 6 | 0.6 |
| 20260210 | 20260211 | 10 | 4 | 0.4 |
| 20260211 | 20260212 | 10 | 7 | 0.7 |
| 20260212 | 20260213 | 10 | 3 | 0.3 |
| 20260213 | 20260224 | 10 | 3 | 0.3 |
| 20260224 | 20260225 | 10 | 5 | 0.5 |
| 20260225 | 20260226 | 10 | 4 | 0.4 |
| 20260226 | 20260227 | 10 | 4 | 0.4 |
| 20260227 | 20260302 | 10 | 1 | 0.1 |

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：1614
- 过滤后行数：1497
- 丢弃全零特征行：29
- 日期总数：22
- 使用日期：22

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：1497
- pos/neg：267/1230
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
