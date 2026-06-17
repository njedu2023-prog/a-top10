#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a_top10.intraday_features import merge_intraday_to_candidates
from a_top10.steps.step6_final_topn import run_step6_final_topn


def main() -> int:
    candidates = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "A", "StrengthScore": 80, "ThemeBoost": 0.8, "Probability": 0.7},
            {"ts_code": "000002.SZ", "name": "B", "StrengthScore": 78, "ThemeBoost": 0.7, "Probability": 0.69},
        ]
    )
    intraday = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "limitup_quality_score": 0.9,
                "intraday_risk_score": 0.1,
                "late_withdraw_score": 0.0,
                "reseal_score": 0.9,
                "open_board_count": 0,
                "seal_stability_score": 0.9,
            },
            {
                "ts_code": "000002.SZ",
                "limitup_quality_score": 0.2,
                "intraday_risk_score": 0.9,
                "late_withdraw_score": 0.85,
                "reseal_score": 0.2,
                "open_board_count": 5,
                "seal_stability_score": 0.2,
            },
        ]
    )
    auction = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "auction_strength_score": 0.85, "auction_real_volume_score": 0.8},
            {"ts_code": "000002.SZ", "auction_strength_score": 0.2, "auction_real_volume_score": 0.2},
        ]
    )

    merged = merge_intraday_to_candidates(candidates, intraday, auction, defaults={})
    result = run_step6_final_topn(merged)
    full = result["full"]

    required = {
        "final_score_base",
        "intraday_bonus",
        "intraday_risk_penalty",
        "final_score_v2",
        "risk_level",
        "risk_tags",
    }
    missing = sorted(required - set(full.columns))
    assert not missing, f"missing Step6 columns: {missing}"

    risky = full[full["ts_code"] == "000002.SZ"].iloc[0]
    assert float(risky["intraday_risk_penalty"]) > 0
    assert float(risky["final_score_v2"]) < float(risky["final_score_base"])
    assert "尾盘撤退" in str(risky["risk_tags"])

    missing_file = merge_intraday_to_candidates(candidates, pd.DataFrame(), pd.DataFrame(), defaults={})
    assert set(missing_file["intraday_data_status"]) == {"missing_file"}
    assert float(missing_file["intraday_risk_score"].max()) == 0.0

    print(json.dumps({"ok": True, "rows": len(full)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
