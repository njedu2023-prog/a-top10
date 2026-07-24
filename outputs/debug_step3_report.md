# Step3 Debug Report

- trade_date: `20260724`
- rows: 41
- snapshot_dir: `_warehouse/a-share-top3-data/data/raw/2026/20260724`
- snapshot_missing: `False`

## Files rows
- daily.csv: 5526
- daily_basic.csv: 5526
- top_list.csv: 79
- moneyflow_hsgt.csv: 1
- limit_list_d.csv: 41
- limit_break_d.csv: 0
- stk_limit.csv: 7707

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
- open_times: 0.5366
- intraday_available: 1.0000
- auction_available: 1.0000

## Strength quality distribution
- A: 41

## Duplicate contract columns after closeout
- amount_x
- total_mv_x
- amount_y
- total_mv_y

```json
{
  "trade_date": "20260724",
  "step": "step3_strength_score_v3_closeout",
  "snapshot_dir": "_warehouse/a-share-top3-data/data/raw/2026/20260724",
  "files": {
    "daily.csv": 5526,
    "daily_basic.csv": 5526,
    "top_list.csv": 79,
    "moneyflow_hsgt.csv": 1,
    "limit_list_d.csv": 41,
    "limit_break_d.csv": 0,
    "stk_limit.csv": 7707
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
    "open_times": 0.5365853658536586,
    "intraday_available": 1.0,
    "auction_available": 1.0
  },
  "quality_distribution": {
    "A": 41
  },
  "duplicate_contract_cols_after_closeout": [
    "amount_x",
    "total_mv_x",
    "amount_y",
    "total_mv_y"
  ]
}
```