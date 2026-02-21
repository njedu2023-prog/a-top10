# Step7 自学习报告（latest）

- 生成时间：2026-02-22 00:51:16
- TopN：10
- Lookback：150 天

## 1) 最新命中

- trade_date：20260212
- next_trade_date：20260213
- hit/topn：4/10
- hit_rate：0.4

## 1.5) Auto-Sampling 状态

- sampling_stage：S1_MVP
- target_rows_per_day：200
- days_covered：13
- quality_gate_pass：True
- pseudo_ratio：0.09487459105779716

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：917
- 过滤后行数：833
- 丢弃全零特征行：29
- 日期总数：13
- 使用日期：13

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：833
- pos/neg：150/683
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
