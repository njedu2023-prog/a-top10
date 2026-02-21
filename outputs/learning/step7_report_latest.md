# Step7 自学习报告（latest）

- 生成时间：2026-02-22 00:17:40
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
- days_covered：6
- quality_gate_pass：True
- pseudo_ratio：0.4713114754098361

## 2) 训练数据概况

- 特征历史文件：outputs/learning/feature_history.csv
- 原始行数：488
- 过滤后行数：356
- 丢弃全零特征行：77
- 日期总数：6
- 使用日期：6

## 3) 训练执行结果

- trained：True
- lr_saved：True
- lgbm_saved：True
- models_dir：models
- train_rows：356
- pos/neg：64/292
- reason：ok
- lr_path：models/step5_lr.joblib
- lgbm_path：models/step5_lgbm.joblib
