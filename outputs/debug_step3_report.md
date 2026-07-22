# Step3 Debug Report

- trade_date: `20260722`
- rows: 46
- snapshot_dir: `_warehouse/a-share-top3-data/data/raw/2026/20260722`
- snapshot_missing: `False`

## Files rows
- daily.csv: 5526
- daily_basic.csv: 5526
- top_list.csv: 87
- moneyflow_hsgt.csv: 1
- limit_list_d.csv: 47
- limit_break_d.csv: 0
- stk_limit.csv: 7705

## Missing rate
- close: 0.0000
- pct_chg: 0.0000
- amount: 1.0000
- turnover_rate: 0.0000
- seal_amount: 0.0000
- open_times: 0.0000
- limit_type: 0.0000
- up_limit: 0.0000
- down_limit: 0.0000
- is_limit_up_pool: 0.0000
- StrengthScore: 0.0000
- limit_strength_raw: 0.0000
- intraday_quality_score: 0.0000
- intraday_soft_risk_score: 0.0000
- strength_plus_score: 0.0000

## Nonnull rate
- close: 1.0000
- turnover_rate: 1.0000
- seal_amount: 1.0000
- open_times: 1.0000
- limit_type: 1.0000
- up_limit: 1.0000
- StrengthScore: 1.0000
- intraday_quality_score: 1.0000
- intraday_soft_risk_score: 1.0000
- strength_plus_score: 1.0000

## Nonzero rate
- StrengthScore: 1.0000
- limit_strength_raw: 1.0000
- open_times: 0.6087
- intraday_available: 0.4565
- auction_available: 0.9783

## Strength quality distribution
- A: 46

## Duplicate contract columns after closeout
- amount_x
- total_mv_x
- amount_y
- total_mv_y

```json
{
  "trade_date": "20260722",
  "step": "step3_strength_score_v3_closeout",
  "snapshot_dir": "_warehouse/a-share-top3-data/data/raw/2026/20260722",
  "files": {
    "daily.csv": 5526,
    "daily_basic.csv": 5526,
    "top_list.csv": 87,
    "moneyflow_hsgt.csv": 1,
    "limit_list_d.csv": 47,
    "limit_break_d.csv": 0,
    "stk_limit.csv": 7705
  },
  "missing_rate": {
    "close": 0.0,
    "pct_chg": 0.0,
    "amount": 1.0,
    "turnover_rate": 0.0,
    "seal_amount": 0.0,
    "open_times": 0.0,
    "limit_type": 0.0,
    "up_limit": 0.0,
    "down_limit": 0.0,
    "is_limit_up_pool": 0.0,
    "StrengthScore": 0.0,
    "limit_strength_raw": 0.0,
    "intraday_quality_score": 0.0,
    "intraday_soft_risk_score": 0.0,
    "strength_plus_score": 0.0
  },
  "nonnull_rate": {
    "close": 1.0,
    "turnover_rate": 1.0,
    "seal_amount": 1.0,
    "open_times": 1.0,
    "limit_type": 1.0,
    "up_limit": 1.0,
    "StrengthScore": 1.0,
    "intraday_quality_score": 1.0,
    "intraday_soft_risk_score": 1.0,
    "strength_plus_score": 1.0
  },
  "nonzero_rate": {
    "StrengthScore": 1.0,
    "limit_strength_raw": 1.0,
    "open_times": 0.6086956521739131,
    "intraday_available": 0.45652173913043476,
    "auction_available": 0.9782608695652174
  },
  "quality_distribution": {
    "A": 46
  },
  "duplicate_contract_cols_after_closeout": [
    "amount_x",
    "total_mv_x",
    "amount_y",
    "total_mv_y"
  ]
}
```