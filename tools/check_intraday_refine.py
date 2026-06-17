#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from a_top10.config import is_a_share_trading_day, prev_a_share_trading_day


REQUIRED_AUDIT_COLUMNS = {
    "rank",
    "ts_code",
    "name",
    "intraday_available",
    "intraday_status",
    "intraday_missing_reason",
    "limitup_quality_score",
    "intraday_quality_score",
    "intraday_soft_risk_score",
    "intraday_hard_risk_flag",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "intraday_confidence_score",
    "risk_level",
    "risk_label",
    "raw_final_score",
    "intraday_bonus",
    "intraday_total_penalty",
    "final_score_v2",
}

REQUIRED_MARKDOWN_TEXT = [
    "分时增强数据覆盖率",
    "分时质量",
    "软风险",
    "硬风险",
    "尾盘风险",
    "回封分",
    "炸板数",
    "竞价强度",
    "覆盖",
    "风险级别",
    "风险标签",
    "高风险个股提示",
]

REQUIRED_PRED_SOURCE_COLUMNS = {
    "trade_date",
    "verify_date",
    "rank",
    "ts_code",
    "name",
    "prob",
    "StrengthScore",
    "ThemeBoost",
    "board",
    "final_score",
    "raw_final_score",
    "final_score_v2",
    "intraday_available",
    "intraday_status",
    "intraday_missing_reason",
    "intraday_quality_score",
    "intraday_soft_risk_score",
    "intraday_hard_risk_flag",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "intraday_confidence_score",
    "risk_level",
    "risk_label",
}


def _latest_trade_date(outputs: Path) -> str:
    files = sorted(outputs.glob("intraday_audit_*.csv"))
    if not files:
        raise FileNotFoundError(f"no intraday_audit_*.csv under {outputs}")
    return files[-1].stem.replace("intraday_audit_", "")


def _normalize_trade_date(td: str) -> str:
    td = str(td or "").strip()
    if len(td) != 8 or not td.isdigit():
        raise ValueError(f"invalid trade_date: {td}")
    if is_a_share_trading_day(td):
        return td
    return prev_a_share_trading_day(td)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--trade-date", default="")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    trade_date = _normalize_trade_date(args.trade_date.strip()) if args.trade_date.strip() else _latest_trade_date(outputs)
    audit_path = outputs / f"intraday_audit_{trade_date}.csv"
    debug_path = outputs / f"debug_intraday_{trade_date}.json"
    md_path = outputs / f"predict_top10_{trade_date}.md"
    pred_source_path = outputs / "decisio" / "pred_source_latest.csv"

    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    if not debug_path.exists():
        raise FileNotFoundError(debug_path)
    if not md_path.exists():
        raise FileNotFoundError(md_path)
    if not pred_source_path.exists():
        raise FileNotFoundError(pred_source_path)

    df = pd.read_csv(audit_path)
    missing_cols = sorted(REQUIRED_AUDIT_COLUMNS - set(df.columns))
    assert not missing_cols, f"missing audit columns: {missing_cols}"

    for col in [
        "limitup_quality_score",
        "intraday_quality_score",
        "intraday_soft_risk_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "auction_strength_score",
        "intraday_confidence_score",
        "final_score_v2",
    ]:
        s = pd.to_numeric(df[col], errors="coerce")
        assert s.dropna().between(0, 1).all(), f"{col} out of 0..1"

    missing = df[pd.to_numeric(df["intraday_available"], errors="coerce").fillna(0) == 0]
    if not missing.empty:
        hard = pd.to_numeric(missing["intraday_hard_risk_flag"], errors="coerce").fillna(0)
        assert not (hard == 1).any(), "missing intraday rows must not be hard risk"
        assert not missing["risk_level"].astype(str).isin(["高", "极高"]).any(), "missing intraday rows must not be high risk"

    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    for key in ["candidate_count", "intraday_rows", "matched_count", "matched_rate", "risk_counts", "default_value_counts"]:
        assert key in debug, f"missing debug key: {key}"

    md = md_path.read_text(encoding="utf-8")
    missing_text = [x for x in REQUIRED_MARKDOWN_TEXT if x not in md]
    assert not missing_text, f"missing markdown text: {missing_text}"

    pred = pd.read_csv(pred_source_path)
    pred_missing = sorted(REQUIRED_PRED_SOURCE_COLUMNS - set(pred.columns))
    assert not pred_missing, f"missing pred_source_latest columns: {pred_missing}"

    print(json.dumps({"ok": True, "trade_date": trade_date, "rows": int(len(df))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
