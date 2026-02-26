# Step3 Debug Report

- trade_date: `20260122`
- rows: 75
- snapshot_dir: `_warehouse/a-share-top3-data/data/raw/2026/20260122`
- snapshot_missing: `False`

## Files rows
- daily.csv: 5466
- daily_basic.csv: 5466
- top_list.csv: 68
- moneyflow_hsgt.csv: 1
- limit_list_d.csv: 75
- limit_break_d.csv: 0

## Missing rate
- pct_chg: 0.0000
- amount: 0.0000
- turnover_rate: 0.0000
- circ_mv: 0.0000
- volume_ratio: 0.0000
- lhb_net_amount: 0.0000
- hsgt_net_amount: 0.0000
- seal_amount: 0.0000
- open_times: 0.0000

## Nonzero rate (关键验收)
- StrengthScore: 1.0000
- turnover_rate: 1.0000
- seal_amount: 0.9733
- open_times: 0.4667

```json
{
  "trade_date": "20260122",
  "snapshot_dir": "_warehouse/a-share-top3-data/data/raw/2026/20260122",
  "files": {
    "daily.csv": 5466,
    "daily_basic.csv": 5466,
    "top_list.csv": 68,
    "moneyflow_hsgt.csv": 1,
    "limit_list_d.csv": 75,
    "limit_break_d.csv": 0
  },
  "missing_rate": {
    "pct_chg": 0.0,
    "amount": 0.0,
    "turnover_rate": 0.0,
    "circ_mv": 0.0,
    "volume_ratio": 0.0,
    "lhb_net_amount": 0.0,
    "hsgt_net_amount": 0.0,
    "seal_amount": 0.0,
    "open_times": 0.0
  },
  "nonzero_rate": {
    "StrengthScore": 1.0,
    "turnover_rate": 1.0,
    "seal_amount": 0.9733333333333334,
    "open_times": 0.4666666666666667
  }
}
```