from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


CODE_COL_CANDIDATES = ["ts_code", "TS_CODE", "code", "CODE", "证券代码", "股票代码", "代码"]
SOURCE_DATE_COLS = ["trade_date", "date", "source_date", "数据日期", "交易日期"]
VERSION_COLS = ["feature_version", "version", "intraday_feature_version", "特征版本"]

INTRADAY_DEFAULTS: Dict[str, Any] = {
    "intraday_available": 0,
    "auction_available": 0,
    "limitup_quality_score": 0.55,
    "intraday_quality_score": 0.55,
    "intraday_soft_risk_score": 0.00,
    "intraday_hard_risk_flag": 0,
    "intraday_risk_score": 0.00,
    "late_withdraw_score": 0.00,
    "reseal_score": 0.50,
    "open_board_count": 0,
    "open_board_risk_score": 0.00,
    "auction_strength_score": 0.50,
    "auction_real_volume_score": 0.50,
    "seal_stability_score": 0.50,
    "intraday_confidence_score": 0.00,
    "weak_reseal_risk_score": 0.00,
    "auction_fake_strength_score": 0.00,
    "volatility_risk_score": 0.00,
    "intraday_bonus": 0.00,
    "intraday_risk_penalty": 0.00,
    "intraday_soft_risk_penalty": 0.00,
    "intraday_hard_risk_penalty": 0.00,
    "intraday_total_penalty": 0.00,
}

STATUS_DEFAULTS = {
    "intraday_status": "missing_file",
    "intraday_data_status": "missing_file",
    "intraday_missing_reason": "intraday_features.csv missing",
    "intraday_source_date": "",
    "intraday_feature_version": "intraday_refine_v2",
    "intraday_matched_key": "",
    "auction_data_status": "missing_file",
    "risk_level": "中",
    "risk_label": "分时缺失",
    "risk_tags": "分时缺失",
}

DEFAULT_FLAG_COLS = [
    "limitup_quality_is_default",
    "reseal_is_default",
    "auction_strength_is_default",
    "late_withdraw_is_default",
    "open_board_count_is_default",
    "intraday_risk_is_default",
]

INTRADAY_FEATURE_COLS = [
    "intraday_available",
    "intraday_status",
    "intraday_missing_reason",
    "intraday_source_date",
    "intraday_feature_version",
    "intraday_matched_key",
    "limitup_quality_score",
    "intraday_quality_score",
    "intraday_soft_risk_score",
    "intraday_hard_risk_flag",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "open_board_risk_score",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_confidence_score",
    "risk_level",
    "risk_label",
    "risk_tags",
    "intraday_data_status",
    "auction_data_status",
    *DEFAULT_FLAG_COLS,
]

ML_INTRADAY_FEATURES = [
    "limitup_quality_score",
    "intraday_quality_score",
    "intraday_soft_risk_score",
    "intraday_hard_risk_flag",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "open_board_risk_score",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_confidence_score",
    "strength_plus_score",
    "intraday_available",
    "auction_available",
]


def get_intraday_cfg(settings: Any) -> Any:
    return getattr(settings, "intraday", None)


def get_intraday_defaults(settings: Any = None) -> Dict[str, Any]:
    cfg = get_intraday_cfg(settings)
    defaults = dict(INTRADAY_DEFAULTS)
    neutral = getattr(cfg, "neutral_values", {}) if cfg is not None else {}
    if isinstance(neutral, dict):
        mapping = {
            "quality": "limitup_quality_score",
            "reseal": "reseal_score",
            "auction_strength": "auction_strength_score",
            "soft_risk": "intraday_soft_risk_score",
            "late_withdraw": "late_withdraw_score",
            "open_board_count": "open_board_count",
            "confidence": "intraday_confidence_score",
        }
        for src, dst in mapping.items():
            if src in neutral:
                defaults[dst] = neutral[src]
    raw = getattr(cfg, "defaults", {}) if cfg is not None else {}
    if isinstance(raw, dict):
        defaults.update(raw)
    return defaults


def get_intraday_dict(settings: Any, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_intraday_cfg(settings)
    raw = getattr(cfg, key, None) if cfg is not None else None
    if isinstance(raw, dict):
        return dict(raw)
    return dict(default or {})


def get_hard_risk_rules(settings: Any) -> Dict[str, Any]:
    rules = get_intraday_dict(settings, "hard_risk_rules")
    if not rules:
        old = get_intraday_dict(settings, "hard_filters")
        if old:
            rules = {
                "enable": old.get("enable", True),
                "late_withdraw_score_gte": old.get("max_late_withdraw_score", 0.80),
                "intraday_soft_risk_gte": old.get("max_intraday_risk_score", 0.85),
                "open_board_count_gte": old.get("max_open_board_count", 4),
                "limitup_quality_lte": old.get("min_limitup_quality_score", 0.25),
                "reseal_score_lte": old.get("min_reseal_score", 0.25),
            }
    return rules


def normalize_ts_code_value(x: Any) -> str:
    s = "" if x is None else str(x).strip().upper()
    if not s or s.lower() in {"nan", "<na>", "none"}:
        return ""
    s = s.replace(" ", "")
    if "." in s:
        left, right = s.split(".", 1)
        left = "".join(ch for ch in left if ch.isdigit())
        right = "".join(ch for ch in right if ch.isalpha())
        if len(left) == 6 and right in {"SH", "SZ", "BJ"}:
            return f"{left}.{right}"
        if len(left) == 6:
            return _suffix_code(left)
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) != 6:
        return ""
    return _suffix_code(digits)


def _suffix_code(code6: str) -> str:
    if code6.startswith(("6", "9")):
        return f"{code6}.SH"
    if code6.startswith(("8", "4")):
        return f"{code6}.BJ"
    return f"{code6}.SZ"


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _raw_float(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="float64")
    if col is None or col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def clip01_series(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype("float64")
    arr = np.asarray(s, dtype=float)
    return pd.Series(np.clip(arr, 0.0, 1.0), dtype="float64")


def _prepare_base(df: pd.DataFrame, available_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", available_col])
    out = df.copy()
    code_col = first_existing_col(out, CODE_COL_CANDIDATES)
    if code_col is None:
        return pd.DataFrame(columns=["ts_code", available_col])
    out["ts_code"] = out[code_col].map(normalize_ts_code_value)
    out = out[out["ts_code"] != ""].copy()
    out[available_col] = 1
    out["_matched_key"] = out["ts_code"]
    return out.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)


def _set_score_with_flag(
    out: pd.DataFrame,
    target: str,
    aliases: Iterable[str],
    default: float,
    flag_col: Optional[str] = None,
    clip01: bool = True,
) -> None:
    source_col = target if target in out.columns else None
    if source_col is None:
        for c in aliases:
            if c in out.columns:
                source_col = c
                break
    raw = _raw_float(out, source_col)
    if flag_col:
        out[flag_col] = raw.isna().astype(int)
        out[f"{target}_raw"] = raw
    filled = raw.fillna(float(default)).astype("float64")
    out[target] = clip01_series(filled) if clip01 else filled


def prepare_intraday_features(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    defaults = {**INTRADAY_DEFAULTS, **(defaults or {})}
    out = _prepare_base(df, "intraday_available")
    if out.empty:
        return out

    _set_score_with_flag(
        out,
        "limitup_quality_score",
        ["limit_up_quality_score", "limitup_quality", "涨停质量"],
        defaults["limitup_quality_score"],
        "limitup_quality_is_default",
    )
    _set_score_with_flag(
        out,
        "intraday_risk_score",
        ["risk_score", "分时风险"],
        defaults["intraday_risk_score"],
        "intraday_risk_is_default",
    )
    _set_score_with_flag(
        out,
        "late_withdraw_score",
        ["tail_withdraw_score", "late_withdraw", "尾盘撤退", "尾盘风险"],
        defaults["late_withdraw_score"],
        "late_withdraw_is_default",
    )
    _set_score_with_flag(
        out,
        "reseal_score",
        ["reseal_quality_score", "reseal_quality", "回封分"],
        defaults["reseal_score"],
        "reseal_is_default",
    )
    _set_score_with_flag(
        out,
        "open_board_count",
        ["open_times", "open_board_times", "open_board_count", "炸板次数", "开板次数"],
        defaults["open_board_count"],
        "open_board_count_is_default",
        clip01=False,
    )
    _set_score_with_flag(out, "seal_stability_score", ["seal_stability", "封单稳定"], defaults["seal_stability_score"])
    _set_score_with_flag(
        out,
        "auction_strength_score",
        ["auction_score", "auction_strength", "竞价强度"],
        defaults["auction_strength_score"],
        "auction_strength_is_default",
    )
    _set_score_with_flag(out, "auction_real_volume_score", ["auction_volume_score", "auction_real_volume", "竞价真实量"], defaults["auction_real_volume_score"])
    _set_score_with_flag(out, "volatility_risk_score", ["volatility_risk", "volatility_risk_score", "波动风险"], defaults["volatility_risk_score"])

    out["open_board_count"] = pd.to_numeric(out["open_board_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["open_board_risk_score"] = (out["open_board_count"] / 5.0).clip(0.0, 1.0)
    out["weak_reseal_risk_score"] = (1.0 - pd.to_numeric(out["reseal_score"], errors="coerce").fillna(0.5)).clip(0.0, 1.0)
    auction = pd.to_numeric(out["auction_strength_score"], errors="coerce").fillna(0.5)
    auction_real = pd.to_numeric(out["auction_real_volume_score"], errors="coerce").fillna(0.5)
    out["auction_fake_strength_score"] = (auction - auction_real).clip(lower=0.0, upper=1.0)

    source_date_col = first_existing_col(out, SOURCE_DATE_COLS)
    version_col = first_existing_col(out, VERSION_COLS)
    out["intraday_source_date"] = out[source_date_col].astype(str) if source_date_col else ""
    out["intraday_feature_version"] = out[version_col].astype(str) if version_col else STATUS_DEFAULTS["intraday_feature_version"]
    out["intraday_matched_key"] = out["_matched_key"]
    out["intraday_status"] = "ok"
    out["intraday_data_status"] = "ok"
    out["intraday_missing_reason"] = ""
    return out


def prepare_auction_features(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    defaults = {**INTRADAY_DEFAULTS, **(defaults or {})}
    out = _prepare_base(df, "auction_available")
    if out.empty:
        return out
    _set_score_with_flag(out, "auction_strength_score", ["auction_score", "auction_imbalance_score", "竞价强度"], defaults["auction_strength_score"])
    _set_score_with_flag(out, "auction_real_volume_score", ["auction_volume_score", "auction_real_volume", "竞价真实量"], defaults["auction_real_volume_score"])
    out["auction_data_status"] = "ok"
    cols = ["ts_code", "auction_available", "auction_strength_score", "auction_real_volume_score", "auction_data_status"]
    return out[cols].drop_duplicates("ts_code", keep="last")


def calc_intraday_quality_score(df: pd.DataFrame, weights: Dict[str, Any]) -> pd.Series:
    w = {
        "limitup_quality_score": 0.30,
        "reseal_score": 0.20,
        "auction_strength_score": 0.15,
        "late_stability_score": 0.20,
        "path_smoothness_score": 0.15,
    }
    if isinstance(weights, dict):
        aliases = {
            "limitup_quality": "limitup_quality_score",
            "auction_strength": "auction_strength_score",
            "late_withdraw_inverse": "late_stability_score",
            "seal_stability": "path_smoothness_score",
        }
        for k, v in weights.items():
            w[aliases.get(k, k)] = v
    raw = (
        float(w["limitup_quality_score"]) * pd.to_numeric(df.get("limitup_quality_score", 0.55), errors="coerce").fillna(0.55)
        + float(w["reseal_score"]) * pd.to_numeric(df.get("reseal_score", 0.5), errors="coerce").fillna(0.5)
        + float(w["auction_strength_score"]) * pd.to_numeric(df.get("auction_strength_score", 0.5), errors="coerce").fillna(0.5)
        + float(w["late_stability_score"]) * (1.0 - pd.to_numeric(df.get("late_withdraw_score", 0.0), errors="coerce").fillna(0.0))
        + float(w["path_smoothness_score"]) * (1.0 - pd.to_numeric(df.get("open_board_risk_score", 0.0), errors="coerce").fillna(0.0))
    )
    return clip01_series(raw)


def calc_intraday_soft_risk_score(df: pd.DataFrame, weights: Dict[str, Any]) -> pd.Series:
    w = {
        "late_withdraw_score": 0.30,
        "open_board_risk_score": 0.25,
        "weak_reseal_risk_score": 0.20,
        "auction_fake_strength_score": 0.15,
        "volatility_risk_score": 0.10,
    }
    if isinstance(weights, dict):
        w.update(weights)
    raw = (
        float(w["late_withdraw_score"]) * pd.to_numeric(df.get("late_withdraw_score", 0), errors="coerce").fillna(0)
        + float(w["open_board_risk_score"]) * pd.to_numeric(df.get("open_board_risk_score", 0), errors="coerce").fillna(0)
        + float(w["weak_reseal_risk_score"]) * pd.to_numeric(df.get("weak_reseal_risk_score", 0), errors="coerce").fillna(0)
        + float(w["auction_fake_strength_score"]) * pd.to_numeric(df.get("auction_fake_strength_score", 0), errors="coerce").fillna(0)
        + float(w["volatility_risk_score"]) * pd.to_numeric(df.get("volatility_risk_score", 0), errors="coerce").fillna(0)
    )
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    return clip01_series(raw).where(available > 0, 0.0)


def calc_intraday_confidence_score(df: pd.DataFrame) -> pd.Series:
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    flag_cols = [c for c in DEFAULT_FLAG_COLS if c in df.columns]
    if not flag_cols:
        return pd.Series(np.where(available > 0, 0.75, 0.0), index=df.index, dtype="float64")
    default_sum = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(1) for c in flag_cols}).sum(axis=1)
    present_ratio = 1.0 - (default_sum / max(1, len(flag_cols)))
    score = np.select(
        [
            available <= 0,
            present_ratio >= 0.95,
            present_ratio >= 0.65,
            present_ratio > 0.0,
        ],
        [0.0, 1.0, 0.75, 0.30],
        default=0.0,
    )
    return pd.Series(score, index=df.index, dtype="float64")


def calc_strength_plus_score(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    base_w = float(cfg.get("base_strength_weight", 0.80)) if isinstance(cfg, dict) else 0.80
    intra_w = float(cfg.get("intraday_quality_weight", 0.20)) if isinstance(cfg, dict) else 0.20
    denom = max(1e-12, base_w + intra_w)
    base_w /= denom
    intra_w /= denom
    base = pd.to_numeric(df.get("base_strength_score", df.get("StrengthScore", 0.0)), errors="coerce").fillna(0.0) / 100.0
    quality = pd.to_numeric(df.get("intraday_quality_score", 0.55), errors="coerce").fillna(0.55)
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    plus01 = base.where(available <= 0, base_w * base + intra_w * quality)
    return clip01_series(plus01) * 100.0


def merge_intraday_to_candidates(
    candidates: pd.DataFrame,
    intraday: pd.DataFrame,
    auction: pd.DataFrame,
    defaults: Dict[str, Any],
    weights: Optional[Dict[str, Any]] = None,
    soft_risk_weights: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    defaults = {**INTRADAY_DEFAULTS, **(defaults or {})}
    out = candidates.copy()
    if "ts_code" not in out.columns:
        return ensure_intraday_columns(out, defaults)
    out["ts_code"] = out["ts_code"].map(normalize_ts_code_value)

    intraday_prepared = prepare_intraday_features(intraday, defaults)
    auction_prepared = prepare_auction_features(auction, defaults)
    intraday_present = intraday is not None and not intraday.empty
    intraday_mergeable = not intraday_prepared.empty
    auction_present = auction is not None and not auction.empty
    auction_mergeable = not auction_prepared.empty

    if intraday_mergeable:
        out = out.merge(intraday_prepared, on="ts_code", how="left", suffixes=("", "_intraday"))
    if auction_mergeable:
        out = out.merge(auction_prepared, on="ts_code", how="left", suffixes=("", "_auction"))
        for c in ["auction_available", "auction_strength_score", "auction_real_volume_score", "auction_data_status"]:
            ac = f"{c}_auction"
            if ac in out.columns:
                if c in out.columns:
                    out[c] = out[c].combine_first(out[ac])
                else:
                    out[c] = out[ac]
                out = out.drop(columns=[ac], errors="ignore")

    out = ensure_intraday_columns(out, defaults)
    available = pd.to_numeric(out["intraday_available"], errors="coerce").fillna(0)
    auction_available = pd.to_numeric(out["auction_available"], errors="coerce").fillna(0)

    if not intraday_present:
        status = "missing_file"
        reason = "intraday_features.csv missing"
    elif not intraday_mergeable:
        status = "missing_columns"
        reason = "intraday_features.csv missing ts_code/code column"
    else:
        status = "missing_stock"
        reason = "stock not covered by intraday_features.csv"

    out["intraday_status"] = np.where(available > 0, "ok", status)
    out["intraday_data_status"] = out["intraday_status"]
    out["intraday_missing_reason"] = np.where(available > 0, "", reason)
    out["intraday_matched_key"] = np.where(available > 0, out["ts_code"], "")
    out["auction_data_status"] = np.where(
        not auction_present,
        "missing_file",
        np.where(auction_available > 0, "ok", "missing_stock"),
    )

    for c in DEFAULT_FLAG_COLS:
        out[c] = pd.to_numeric(out.get(c, 1), errors="coerce").fillna(1).astype(int)
        out.loc[available <= 0, c] = 1

    out["open_board_risk_score"] = (pd.to_numeric(out["open_board_count"], errors="coerce").fillna(0) / 5.0).clip(0.0, 1.0)
    out["weak_reseal_risk_score"] = (1.0 - pd.to_numeric(out["reseal_score"], errors="coerce").fillna(0.5)).clip(0.0, 1.0)
    out["auction_fake_strength_score"] = (
        pd.to_numeric(out["auction_strength_score"], errors="coerce").fillna(0.5)
        - pd.to_numeric(out["auction_real_volume_score"], errors="coerce").fillna(0.5)
    ).clip(lower=0.0, upper=1.0)
    out["intraday_quality_score"] = calc_intraday_quality_score(out, weights or {})
    out["intraday_soft_risk_score"] = calc_intraday_soft_risk_score(out, soft_risk_weights or {})
    raw_risk = pd.to_numeric(out.get("intraday_risk_score_raw", out.get("intraday_risk_score", 0)), errors="coerce").fillna(0)
    out["intraday_risk_score"] = clip01_series(np.maximum(out["intraday_soft_risk_score"], raw_risk * 0.75))
    out["intraday_confidence_score"] = calc_intraday_confidence_score(out)
    return out


def ensure_intraday_columns(df: pd.DataFrame, defaults: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    defaults = {**INTRADAY_DEFAULTS, **(defaults or {})}
    out = df.copy()
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default
    for c, default in STATUS_DEFAULTS.items():
        if c not in out.columns:
            out[c] = default
    for c in DEFAULT_FLAG_COLS:
        if c not in out.columns:
            out[c] = 1
    numeric_cols = [
        "intraday_available",
        "auction_available",
        "limitup_quality_score",
        "intraday_quality_score",
        "intraday_soft_risk_score",
        "intraday_hard_risk_flag",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "open_board_count",
        "open_board_risk_score",
        "auction_strength_score",
        "auction_real_volume_score",
        "seal_stability_score",
        "intraday_confidence_score",
        "weak_reseal_risk_score",
        "auction_fake_strength_score",
        "volatility_risk_score",
        "intraday_bonus",
        "intraday_risk_penalty",
        "intraday_soft_risk_penalty",
        "intraday_hard_risk_penalty",
        "intraday_total_penalty",
        *DEFAULT_FLAG_COLS,
    ]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(float(defaults.get(c, 0.0)))
    for c in [
        "limitup_quality_score",
        "intraday_quality_score",
        "intraday_soft_risk_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "open_board_risk_score",
        "auction_strength_score",
        "auction_real_volume_score",
        "seal_stability_score",
        "intraday_confidence_score",
        "weak_reseal_risk_score",
        "auction_fake_strength_score",
        "volatility_risk_score",
        "intraday_bonus",
        "intraday_risk_penalty",
        "intraday_soft_risk_penalty",
        "intraday_hard_risk_penalty",
        "intraday_total_penalty",
    ]:
        out[c] = clip01_series(out[c])
    out["open_board_count"] = out["open_board_count"].clip(lower=0.0)
    out["intraday_data_status"] = out["intraday_status"]
    if "risk_label" in out.columns and "risk_tags" not in out.columns:
        out["risk_tags"] = out["risk_label"]
    if "risk_tags" in out.columns and "risk_label" not in out.columns:
        out["risk_label"] = out["risk_tags"]
    return out


def calc_intraday_bonus(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    q_w = float(cfg.get("quality_bonus_weight", 0.025)) if isinstance(cfg, dict) else 0.025
    conf_w = float(cfg.get("confidence_bonus_weight", 0.010)) if isinstance(cfg, dict) else 0.010
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    quality = pd.to_numeric(df.get("intraday_quality_score", 0.55), errors="coerce").fillna(0.55)
    confidence = pd.to_numeric(df.get("intraday_confidence_score", 0.0), errors="coerce").fillna(0.0)
    return ((q_w * quality + conf_w * confidence) * (available > 0).astype(float)).astype("float64")


def calc_intraday_soft_risk_penalty(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    risk_w = float(cfg.get("soft_risk_penalty_weight", cfg.get("risk_penalty_weight", 0.045))) if isinstance(cfg, dict) else 0.045
    late_w = float(cfg.get("late_withdraw_penalty_weight", 0.035)) if isinstance(cfg, dict) else 0.035
    open_w = float(cfg.get("open_board_penalty_weight", 0.020)) if isinstance(cfg, dict) else 0.020
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    risk = pd.to_numeric(df.get("intraday_soft_risk_score", 0.0), errors="coerce").fillna(0.0)
    late = pd.to_numeric(df.get("late_withdraw_score", 0.0), errors="coerce").fillna(0.0)
    open_board = pd.to_numeric(df.get("open_board_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    penalty = risk_w * risk + late_w * late + open_w * (open_board / 5.0).clip(0.0, 1.0)
    return (penalty * (available > 0).astype(float)).astype("float64")


def calc_intraday_hard_flags(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    hard = dict(cfg or {})
    enabled = bool(hard.get("enable", True))
    out = df.copy()
    available = pd.to_numeric(out.get("intraday_available", 0), errors="coerce").fillna(0)
    if not enabled:
        out["intraday_hard_risk_flag"] = 0
        return out
    flag = (
        (pd.to_numeric(out.get("late_withdraw_score", 0), errors="coerce").fillna(0) >= float(hard.get("late_withdraw_score_gte", 0.80)))
        | (pd.to_numeric(out.get("intraday_soft_risk_score", 0), errors="coerce").fillna(0) >= float(hard.get("intraday_soft_risk_gte", 0.85)))
        | (pd.to_numeric(out.get("open_board_count", 0), errors="coerce").fillna(0) >= float(hard.get("open_board_count_gte", 4)))
        | (pd.to_numeric(out.get("limitup_quality_score", 0.55), errors="coerce").fillna(0.55) <= float(hard.get("limitup_quality_lte", 0.25)))
        | (pd.to_numeric(out.get("reseal_score", 0.5), errors="coerce").fillna(0.5) <= float(hard.get("reseal_score_lte", 0.25)))
    )
    out["intraday_hard_risk_flag"] = (flag & (available > 0)).astype(int)
    return out


def calc_intraday_hard_risk_penalty(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    hard_penalty = float(cfg.get("hard_risk_penalty", 0.080)) if isinstance(cfg, dict) else 0.080
    flag = pd.to_numeric(df.get("intraday_hard_risk_flag", 0), errors="coerce").fillna(0)
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    return (hard_penalty * (flag > 0).astype(float) * (available > 0).astype(float)).astype("float64")


def calc_intraday_risk_penalty(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    return calc_intraday_soft_risk_penalty(df, cfg) + calc_intraday_hard_risk_penalty(df, cfg)


def build_risk_tags(row: pd.Series) -> str:
    tags = []
    available = float(row.get("intraday_available", 0) or 0) > 0
    if not available:
        tags.append("分时缺失")
    if available and float(row.get("intraday_risk_score", 0) or 0) >= 0.65:
        tags.append("分时高风险")
    if available and float(row.get("late_withdraw_score", 0) or 0) >= 0.70:
        tags.append("尾盘撤退")
    if available and float(row.get("open_board_count", 0) or 0) >= 3:
        tags.append("多次炸板")
    if available and float(row.get("reseal_score", 0.5) or 0.5) <= 0.35:
        tags.append("回封偏弱")
    if available and float(row.get("auction_fake_strength_score", 0) or 0) >= 0.35:
        tags.append("竞价虚强")
    if available and float(row.get("volatility_risk_score", 0) or 0) >= 0.60:
        tags.append("波动异常")
    if float(row.get("intraday_confidence_score", 0) or 0) < 0.50:
        tags.append("低置信")
    if available and not tags:
        tags.append("健康涨停")
    return "|".join(tags)


def build_risk_level(row: pd.Series) -> str:
    available = float(row.get("intraday_available", 0) or 0) > 0
    hard = float(row.get("intraday_hard_risk_flag", 0) or 0) > 0
    soft = float(row.get("intraday_soft_risk_score", 0) or 0)
    if hard and soft >= 0.70:
        return "极高"
    if hard or soft >= 0.65:
        return "高"
    if soft >= 0.30 or not available:
        return "中"
    return "低"


def build_intraday_conclusion(row: pd.Series) -> str:
    tags = build_risk_tags(row)
    if float(row.get("intraday_available", 0) or 0) <= 0:
        return "分时数据缺失，已按中性值降级"
    if tags and tags != "健康涨停":
        return f"分时风险：{tags}"
    return "分时结构健康"


def build_intraday_debug_summary(full_df: pd.DataFrame, topn_df: pd.DataFrame, intraday_input: Dict[str, Any], trade_date: str) -> Dict[str, Any]:
    full = full_df if isinstance(full_df, pd.DataFrame) else pd.DataFrame()
    topn = topn_df if isinstance(topn_df, pd.DataFrame) else pd.DataFrame()
    avail = pd.to_numeric(full.get("intraday_available", pd.Series([], dtype=float)), errors="coerce").fillna(0)
    top_avail = pd.to_numeric(topn.get("intraday_available", pd.Series([], dtype=float)), errors="coerce").fillna(0)
    status = full.get("intraday_status", pd.Series([], dtype="object")).fillna("").astype(str)
    level = full.get("risk_level", pd.Series([], dtype="object")).fillna("").astype(str)
    label = full.get("risk_label", full.get("risk_tags", pd.Series([], dtype="object"))).fillna("").astype(str)
    return {
        "trade_date": str(trade_date),
        "candidate_count": int(len(full)),
        "topn_count": int(len(topn)),
        "intraday_rows": int(intraday_input.get("intraday_rows", intraday_input.get("intraday_features_rows", 0)) or 0),
        "auction_rows": int(intraday_input.get("auction_rows", intraday_input.get("stk_auction_rows", 0)) or 0),
        "matched_count": int((avail > 0).sum()),
        "matched_rate": float((avail > 0).mean()) if len(avail) else 0.0,
        "topn_matched_count": int((top_avail > 0).sum()),
        "topn_matched_rate": float((top_avail > 0).mean()) if len(top_avail) else 0.0,
        "missing_count": int((avail <= 0).sum()),
        "missing_top10_count": int((top_avail <= 0).sum()),
        "missing_reason_counts": status[status != "ok"].value_counts().to_dict(),
        "risk_counts": {
            "low": int((level == "低").sum()),
            "medium": int((level == "中").sum()),
            "high": int((level == "高").sum()),
            "extreme": int((level == "极高").sum()),
        },
        "default_value_counts": {
            "quality_default": int(pd.to_numeric(full.get("limitup_quality_is_default", 0), errors="coerce").fillna(0).sum()) if len(full) else 0,
            "reseal_default": int(pd.to_numeric(full.get("reseal_is_default", 0), errors="coerce").fillna(0).sum()) if len(full) else 0,
            "auction_default": int(pd.to_numeric(full.get("auction_strength_is_default", 0), errors="coerce").fillna(0).sum()) if len(full) else 0,
        },
        "risk_label_counts": label.str.split("|").explode().replace("", np.nan).dropna().value_counts().to_dict() if len(label) else {},
    }
