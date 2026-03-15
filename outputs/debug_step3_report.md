# Step3 Debug Report

- trade_date: `20260303`
- rows: 83
- snapshot_dir: `_warehouse/a-share-top3-data/data/raw/2026/20260303`
- snapshot_missing: `False`

## Files rows
- daily.csv: 5472
- daily_basic.csv: 5472
- top_list.csv: 121
- moneyflow_hsgt.csv: 1
- limit_list_d.csv: 83
- limit_break_d.csv: 0
- stk_limit.csv: 7488

## Missing rate
- close: 0.0000
- pct_chg: 0.0000
- amount: 0.0000
- turnover_rate: 0.0000
- seal_amount: 0.0000
- open_times: 0.0000
- limit_type: 0.0000
- up_limit: 0.0000
- down_limit: 0.0000
- is_limit_up_pool: 0.0000
- StrengthScore: 0.0000
- limit_strength_raw: 0.0000

## Nonnull rate
- close: 1.0000
- turnover_rate: 1.0000
- seal_amount: 1.0000
- open_times: 1.0000
- limit_type: 1.0000
- up_limit: 1.0000
- StrengthScore: 1.0000

## Nonzero rate
- StrengthScore: 1.0000
- limit_strength_raw: 1.0000
- open_times: 0.6506

## Strength quality distribution
- A: 83

## Duplicate contract columns after closeout
- none

```json
{
  "trade_date": "20260303",
  "step": "step3_strength_score_v3_closeout",
  "snapshot_dir": "_warehouse/a-share-top3-data/data/raw/2026/20260303",
  "files": {
    "daily.csv": 5472,
    "daily_basic.csv": 5472,
    "top_list.csv": 121,
    "moneyflow_hsgt.csv": 1,
    "limit_list_d.csv": 83,
    "limit_break_d.csv": 0,
    "stk_limit.csv": 7488
  },
  "missing_rate": {
    "close": 0.0,
    "pct_chg": 0.0,
    "amount": 0.0,
    "turnover_rate": 0.0,
    "seal_amount": 0.0,
    "open_times": 0.0,
    "limit_type": 0.0,
    "up_limit": 0.0,
    "down_limit": 0.0,
    "is_limit_up_pool": 0.0,
    "StrengthScore": 0.0,
    "limit_strength_raw": 0.0
  },
  "nonnull_rate": {
    "close": 1.0,
    "turnover_rate": 1.0,
    "seal_amount": 1.0,
    "open_times": 1.0,
    "limit_type": 1.0,
    "up_limit": 1.0,
    "StrengthScore": 1.0
  },
  "nonzero_rate": {
    "StrengthScore": 1.0,
    "limit_strength_raw": 1.0,
    "open_times": 0.6506024096385542
  },
  "quality_distribution": {
    "A": 83
  },
  "duplicate_contract_cols_after_closeout": []
}
```