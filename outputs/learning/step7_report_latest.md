# Step7 自学习报告（latest）

- 生成时间：2026-03-05 19:59:43
- Today：20260305
- LatestSnapshot：20260305
- LabelUpperBound：20260305
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260304
- expected_next_trade_date：20260305
- actual_next_trade_date：20260305
- hit/topn：3/10
- hit_rate：0.3
- note：src=pred_top10_history

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | actual_next_trade_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260211 | 20260212 | 10 | 7 | 0.7 |
| 20260212 | 20260213 | 10 | 3 | 0.3 |
| 20260213 | 20260224 | 10 | 3 | 0.3 |
| 20260224 | 20260225 | 10 | 5 | 0.5 |
| 20260225 | 20260226 | 10 | 4 | 0.4 |
| 20260226 | 20260227 | 10 | 4 | 0.4 |
| 20260227 | 20260302 | 10 | 1 | 0.1 |
| 20260302 | 20260303 | 10 | 7 | 0.7 |
| 20260303 | 20260304 | 10 | 2 | 0.2 |
| 20260304 | 20260305 | 10 | 3 | 0.3 |

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：1810
- 过滤后行数：1712
- 丢弃全零特征行：29
- 日期总数：25
- 使用日期：25

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：1712
- pos/neg：317/1395
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
