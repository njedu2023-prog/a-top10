# Step7 自学习报告（latest）

- 生成时间：2026-03-11 19:59:22
- Today：20260311
- LatestSnapshot：20260311
- LabelUpperBound：20260311
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260310
- expected_next_trade_date：20260311
- actual_next_trade_date：20260311
- hit/topn：2/10
- hit_rate：0.2
- note：src=pred_top10_history

## 1.1) 近10日 Top10 命中率（done-only）

| trade_date | actual_next_trade_date | topn | hit | hit_rate |
| --- | --- | --- | --- | --- |
| 20260225 | 20260226 | 10 | 4 | 0.4 |
| 20260226 | 20260227 | 10 | 4 | 0.4 |
| 20260227 | 20260302 | 10 | 1 | 0.1 |
| 20260302 | 20260303 | 10 | 7 | 0.7 |
| 20260303 | 20260304 | 10 | 2 | 0.2 |
| 20260304 | 20260305 | 10 | 3 | 0.3 |
| 20260305 | 20260306 | 10 | 3 | 0.3 |
| 20260306 | 20260309 | 10 | 3 | 0.3 |
| 20260309 | 20260310 | 10 | 0 | 0.0 |
| 20260310 | 20260311 | 10 | 2 | 0.2 |

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：2035
- 过滤后行数：1953
- 丢弃全零特征行：29
- 日期总数：29
- 使用日期：29

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：1953
- pos/neg：352/1601
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
