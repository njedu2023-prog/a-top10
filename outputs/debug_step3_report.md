# Step3 Debug Report

- trade_date: `20260313`
- rows: 59
- snapshot_dir: `_warehouse/a-share-top3-data/data/raw/2026/20260313`
- snapshot_missing: `False`

## Files rows
- daily.csv: 5481
- daily_basic.csv: 5481
- top_list.csv: 63
- moneyflow_hsgt.csv: 1
- limit_list_d.csv: 59
- limit_break_d.csv: 0
- stk_limit.csv: 7496

## Missing rate
- pct_chg: 0.0000
- amount: 0.0000
- turnover_rate: 0.0000
- seal_amount: 0.0000
- open_times: 0.0000
- limit_type: 1.0000
- up_limit: 0.0000
- is_limit_up_pool: 0.0000
- StrengthScore: 0.0000
- limit_strength_raw: 0.0000

## Nonnull rate
- turnover_rate: 1.0000
- seal_amount: 1.0000
- open_times: 1.0000
- StrengthScore: 1.0000

## Nonzero rate
- StrengthScore: 1.0000
- limit_strength_raw: 1.0000

## Strength quality distribution
- A: 59

```json
{
  "trade_date": "20260313",
  "step": "step3_strength_score_v3",
  "snapshot_dir": "_warehouse/a-share-top3-data/data/raw/2026/20260313",
  "files": {
    "daily.csv": 5481,
    "daily_basic.csv": 5481,
    "top_list.csv": 63,
    "moneyflow_hsgt.csv": 1,
    "limit_list_d.csv": 59,
    "limit_break_d.csv": 0,
    "stk_limit.csv": 7496
  },
  "missing_rate": {
    "pct_chg": 0.0,
    "amount": 0.0,
    "turnover_rate": 0.0,
    "seal_amount": 0.0,
    "open_times": 0.0,
    "limit_type": 1.0,
    "up_limit": 0.0,
    "is_limit_up_pool": 0.0,
    "StrengthScore": 0.0,
    "limit_strength_raw": 0.0
  },
  "nonnull_rate": {
    "turnover_rate": 1.0,
    "seal_amount": 1.0,
    "open_times": 1.0,
    "StrengthScore": 1.0
  },
  "nonzero_rate": {
    "StrengthScore": 1.0,
    "limit_strength_raw": 1.0
  },
  "quality_distribution": {
    "A": 59
  }
}
```