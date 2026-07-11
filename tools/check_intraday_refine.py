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

MIN_CODE_OVERLAP_RATE = 0.01
MAX_UPPER_SATURATION_RATE = 0.95


class ContractValidationError(AssertionError):
    pass


def _require(condition: bool, reason: str, **details: object) -> None:
    if condition:
        return
    payload = {"reason": reason, **details}
    raise ContractValidationError(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _as_int(debug: dict, key: str) -> int:
    try:
        return int(debug.get(key, 0) or 0)
    except (TypeError, ValueError):
        raise ContractValidationError(
            json.dumps({"reason": "invalid_debug_integer", "key": key, "value": debug.get(key)}, ensure_ascii=False)
        )


def _as_float(debug: dict, key: str) -> float:
    try:
        return float(debug.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        raise ContractValidationError(
            json.dumps({"reason": "invalid_debug_number", "key": key, "value": debug.get(key)}, ensure_ascii=False)
        )


def _validate_distribution(
    name: str,
    distributions: dict,
    nondegenerate: list,
    max_upper_saturation_rate: float,
) -> None:
    _require(bool(nondegenerate), f"{name}_nondegenerate_features_missing")
    for col in nondegenerate:
        stats = distributions.get(col)
        _require(isinstance(stats, dict), f"{name}_distribution_missing", feature=col)
        sample_count = int(stats.get("sample_count", 0) or 0)
        unique_count = int(stats.get("unique_count", 0) or 0)
        upper_saturation_rate = float(stats.get("upper_saturation_rate", 0.0) or 0.0)
        _require(
            sample_count == 1 or unique_count > 1,
            f"{name}_feature_degenerate",
            feature=col,
            sample_count=sample_count,
            unique_count=unique_count,
        )
        if col != "open_board_count" and sample_count >= 2:
            _require(
                upper_saturation_rate < max_upper_saturation_rate,
                f"{name}_feature_upper_saturated",
                feature=col,
                upper_saturation_rate=upper_saturation_rate,
                threshold=max_upper_saturation_rate,
            )


def validate_feature_contract(
    audit: pd.DataFrame,
    debug: dict,
    min_code_overlap_rate: float = MIN_CODE_OVERLAP_RATE,
    max_upper_saturation_rate: float = MAX_UPPER_SATURATION_RATE,
    require_intraday: bool = False,
) -> dict:
    _require(isinstance(debug, dict), "debug_payload_not_object")
    required_debug = {
        "candidate_count",
        "intraday_rows",
        "code_overlap_count",
        "code_overlap_rate",
        "matched_count",
        "matched_rate",
        "valid_feature_count",
        "valid_feature_rate",
        "nondegenerate_features",
        "feature_distributions",
        "auction_rows",
        "auction_code_overlap_count",
        "auction_code_overlap_rate",
        "auction_valid_count",
        "auction_valid_rate",
        "auction_nondegenerate_features",
        "auction_feature_distributions",
    }
    missing_debug = sorted(required_debug - set(debug))
    _require(not missing_debug, "missing_contract_debug_keys", keys=missing_debug)

    candidate_count = _as_int(debug, "candidate_count")
    intraday_rows = _as_int(debug, "intraday_rows")
    code_overlap_count = _as_int(debug, "code_overlap_count")
    code_overlap_rate = _as_float(debug, "code_overlap_rate")
    valid_count = _as_int(debug, "valid_feature_count")
    valid_rate = _as_float(debug, "valid_feature_rate")
    matched_count = _as_int(debug, "matched_count")
    matched_rate = _as_float(debug, "matched_rate")

    _require(candidate_count == len(audit), "candidate_count_mismatch", debug=candidate_count, audit_rows=len(audit))
    _require(0 <= valid_count <= code_overlap_count <= candidate_count, "invalid_intraday_coverage_counts")
    expected_overlap_rate = code_overlap_count / candidate_count if candidate_count else 0.0
    expected_valid_rate = valid_count / candidate_count if candidate_count else 0.0
    _require(abs(code_overlap_rate - expected_overlap_rate) <= 1e-9, "code_overlap_rate_mismatch")
    _require(abs(valid_rate - expected_valid_rate) <= 1e-9, "valid_feature_rate_mismatch")
    _require(matched_count == valid_count and abs(matched_rate - valid_rate) <= 1e-9, "legacy_coverage_alias_mismatch")

    actual_valid = int(pd.to_numeric(audit["intraday_available"], errors="coerce").fillna(0).gt(0).sum())
    _require(actual_valid == valid_count, "audit_valid_count_mismatch", debug=valid_count, audit=actual_valid)

    warnings = []
    if intraday_rows > 0 and candidate_count > 0:
        try:
            _require(code_overlap_count > 0, "intraday_code_overlap_zero", intraday_rows=intraday_rows)
            _require(
                code_overlap_rate >= min_code_overlap_rate,
                "intraday_code_overlap_too_low",
                code_overlap_rate=code_overlap_rate,
                threshold=min_code_overlap_rate,
            )
            _require(valid_count > 0, "intraday_valid_nondegenerate_coverage_zero")
            _validate_distribution(
                "intraday",
                debug.get("feature_distributions", {}),
                list(debug.get("nondegenerate_features", [])),
                max_upper_saturation_rate,
            )
        except ContractValidationError as exc:
            if require_intraday:
                raise
            try:
                reason = json.loads(str(exc)).get("reason", str(exc))
            except json.JSONDecodeError:
                reason = str(exc)
            warnings.append(f"optional intraday enhancement disabled: {reason}")

    auction_rows = _as_int(debug, "auction_rows")
    auction_overlap_count = _as_int(debug, "auction_code_overlap_count")
    auction_overlap_rate = _as_float(debug, "auction_code_overlap_rate")
    auction_valid_count = _as_int(debug, "auction_valid_count")
    auction_valid_rate = _as_float(debug, "auction_valid_rate")
    _require(0 <= auction_valid_count <= auction_overlap_count <= candidate_count, "invalid_auction_coverage_counts")
    expected_auction_overlap_rate = auction_overlap_count / candidate_count if candidate_count else 0.0
    expected_auction_valid_rate = auction_valid_count / candidate_count if candidate_count else 0.0
    _require(abs(auction_overlap_rate - expected_auction_overlap_rate) <= 1e-9, "auction_code_overlap_rate_mismatch")
    _require(abs(auction_valid_rate - expected_auction_valid_rate) <= 1e-9, "auction_valid_rate_mismatch")

    if auction_rows > 0 and candidate_count > 0:
        _require(auction_overlap_count > 0, "auction_code_overlap_zero", auction_rows=auction_rows)
        _require(
            auction_overlap_rate >= min_code_overlap_rate,
            "auction_code_overlap_too_low",
            auction_code_overlap_rate=auction_overlap_rate,
            threshold=min_code_overlap_rate,
        )
        if auction_valid_count > 0:
            _validate_distribution(
                "auction",
                debug.get("auction_feature_distributions", {}),
                list(debug.get("auction_nondegenerate_features", [])),
                max_upper_saturation_rate,
            )
        elif str(debug.get("auction_source", "")) == "auction_features.csv":
            _require(False, "auction_features_valid_nondegenerate_coverage_zero")
        else:
            warnings.append("raw stk_auction.csv has no engineered non-degenerate score; neutral fallback used")

    return {
        "candidate_count": candidate_count,
        "code_overlap_rate": code_overlap_rate,
        "valid_feature_rate": valid_rate,
        "auction_code_overlap_rate": auction_overlap_rate,
        "auction_valid_rate": auction_valid_rate,
        "warnings": warnings,
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


def _run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--trade-date", default="")
    ap.add_argument("--strict-intraday", action="store_true")
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
    _require(not missing_cols, "missing_audit_columns", columns=missing_cols)

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
        _require(bool(s.dropna().between(0, 1).all()), "score_out_of_range", column=col)

    missing = df[pd.to_numeric(df["intraday_available"], errors="coerce").fillna(0) == 0]
    if not missing.empty:
        hard = pd.to_numeric(missing["intraday_hard_risk_flag"], errors="coerce").fillna(0)
        _require(not bool((hard == 1).any()), "missing_intraday_row_marked_hard_risk")
        _require(
            not bool(missing["risk_level"].astype(str).isin(["高", "极高"]).any()),
            "missing_intraday_row_marked_high_risk",
        )

    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    for key in ["candidate_count", "intraday_rows", "matched_count", "matched_rate", "risk_counts", "default_value_counts"]:
        _require(key in debug, "missing_debug_key", key=key)
    contract = validate_feature_contract(df, debug, require_intraday=bool(args.strict_intraday))

    md = md_path.read_text(encoding="utf-8")
    missing_text = [x for x in REQUIRED_MARKDOWN_TEXT if x not in md]
    _require(not missing_text, "missing_markdown_text", text=missing_text)

    pred = pd.read_csv(pred_source_path)
    pred_missing = sorted(REQUIRED_PRED_SOURCE_COLUMNS - set(pred.columns))
    _require(not pred_missing, "missing_pred_source_columns", columns=pred_missing)

    print(json.dumps({"ok": True, "trade_date": trade_date, "rows": int(len(df)), "contract": contract}, ensure_ascii=False))
    return 0


def main() -> int:
    try:
        return _run()
    except ContractValidationError as exc:
        try:
            error = json.loads(str(exc))
        except json.JSONDecodeError:
            error = {"reason": "contract_validation_failed", "message": str(exc)}
        print(json.dumps({"ok": False, "error": error}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
