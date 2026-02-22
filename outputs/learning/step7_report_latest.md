# Step7 自学习报告（latest）

- 生成时间：2026-02-23 02:02:18
- Today：20260223
- LatestSnapshot：20260213
- LabelUpperBound：20260213
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260212
- expected_next_trade_date：20260213
- actual_next_trade_date：20260213
- hit/topn：3/10
- hit_rate：0.3

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：1145
- 过滤后行数：1061
- 丢弃全零特征行：29
- 日期总数：16
- 使用日期：16

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：1061
- pos/neg：189/872
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
