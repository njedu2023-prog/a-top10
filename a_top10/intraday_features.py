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

INTRADAY_CONTRACT_SCORE_COLS = [
    "limitup_quality_score",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "seal_stability_score",
    "volatility_risk_score",
]

AUCTION_CONTRACT_SCORE_COLS = [
    "auction_strength_score",
    "auction_real_volume_score",
]

SCORE_SCALE_EPSILON = 1e-9


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


def normalize_score_series(s: pd.Series) -> pd.Series:
    """Normalize a score column expressed either as 0..1 or 0..100.

    Scale detection is column-level because an upstream score column has one
    contract. Values outside both supported ranges are treated as invalid
    instead of being clipped to a boundary.
    """
    numeric = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    supported = numeric.where(numeric.between(0.0, 100.0, inclusive="both"))
    valid = supported.dropna()
    if valid.empty:
        return supported
    if bool((valid > 1.0 + SCORE_SCALE_EPSILON).any()):
        supported = supported / 100.0
    return supported.where(supported.between(0.0, 1.0, inclusive="both"))


def _normalize_score_default(value: Any, fallback: float = 0.0) -> float:
    normalized = normalize_score_series(pd.Series([value], dtype="object"))
    if normalized.empty or pd.isna(normalized.iloc[0]):
        return float(fallback)
    return float(normalized.iloc[0])


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
    source_col = first_existing_col(out, [target, *list(aliases)])
    raw = _raw_float(out, source_col)
    if clip01:
        normalized = normalize_score_series(raw)
        default_value = _normalize_score_default(default, fallback=0.0)
    else:
        normalized = raw.where(raw >= 0.0)
        default_value = max(0.0, float(default))
    provided = normalized.notna()
    out[f"_{target}_provided"] = provided.astype(int)
    if flag_col:
        out[flag_col] = (~provided).astype(int)
        out[f"{target}_source_raw"] = raw
        # Existing consumers expect *_raw to be on the model's 0..1 scale.
        out[f"{target}_raw"] = normalized
    out[target] = normalized.fillna(default_value).astype("float64")


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
        ["reseal_quality_score", "reseal_quality", "reseal_acceptance_score", "reseal_speed_score", "回封分"],
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
    _set_score_with_flag(
        out,
        "seal_stability_score",
        ["seal_stability", "limitup_path_score", "封单稳定"],
        defaults["seal_stability_score"],
    )
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
    provided_cols = [f"_{c}_provided" for c in [*INTRADAY_CONTRACT_SCORE_COLS, "open_board_count"] if f"_{c}_provided" in out.columns]
    out["_intraday_feature_valid_count"] = out[provided_cols].sum(axis=1) if provided_cols else 0
    out["intraday_available"] = (out["_intraday_feature_valid_count"] > 0).astype(int)
    out["_intraday_code_match"] = 1
    out["intraday_status"] = np.where(out["intraday_available"] > 0, "ok", "invalid_features")
    out["intraday_data_status"] = out["intraday_status"]
    out["intraday_missing_reason"] = np.where(
        out["intraday_available"] > 0,
        "",
        "intraday row has no valid 0..1 or 0..100 score feature",
    )
    return out


def prepare_auction_features(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    defaults = {**INTRADAY_DEFAULTS, **(defaults or {})}
    out = _prepare_base(df, "auction_available")
    if out.empty:
        return out
    _set_score_with_flag(
        out,
        "auction_strength_score",
        ["auction_score", "auction_imbalance_score", "auction_strength", "竞价强度"],
        defaults["auction_strength_score"],
        "auction_strength_is_default",
    )
    _set_score_with_flag(
        out,
        "auction_real_volume_score",
        ["auction_volume_score", "auction_real_volume", "竞价真实量"],
        defaults["auction_real_volume_score"],
    )
    provided_cols = [f"_{c}_provided" for c in AUCTION_CONTRACT_SCORE_COLS]
    out["_auction_feature_valid_count"] = out[provided_cols].sum(axis=1)
    out["auction_available"] = (out["_auction_feature_valid_count"] > 0).astype(int)
    out["_auction_code_match"] = 1
    out["auction_data_status"] = np.where(out["auction_available"] > 0, "ok", "invalid_features")
    cols = [
        "ts_code",
        "auction_available",
        "auction_strength_score",
        "auction_real_volume_score",
        "auction_strength_is_default",
        "auction_data_status",
        "_auction_code_match",
        "_auction_feature_valid_count",
        *provided_cols,
    ]
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


def _prefixed_source(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code"])
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != "ts_code"})


def _nondegenerate_source_features(
    merged: pd.DataFrame,
    prefix: str,
    feature_cols: Sequence[str],
    code_match: pd.Series,
) -> list[str]:
    matched_count = int(code_match.sum())
    nondegenerate: list[str] = []
    for col in feature_cols:
        value_col = f"{prefix}{col}"
        provided_col = f"{prefix}_{col}_provided"
        if value_col not in merged.columns or provided_col not in merged.columns:
            continue
        provided = pd.to_numeric(merged[provided_col], errors="coerce").fillna(0).gt(0)
        values = pd.to_numeric(merged.loc[code_match & provided, value_col], errors="coerce").dropna()
        if values.empty:
            continue
        # A one-row candidate set cannot establish variance, but its valid
        # source value must remain usable. Normal production pools use >1 row.
        if matched_count == 1 or int(values.nunique(dropna=True)) > 1:
            nondegenerate.append(col)
    return nondegenerate


def _row_has_source_feature(
    merged: pd.DataFrame,
    prefix: str,
    feature_cols: Sequence[str],
    code_match: pd.Series,
) -> pd.Series:
    valid = pd.Series(False, index=merged.index, dtype=bool)
    for col in feature_cols:
        provided_col = f"{prefix}_{col}_provided"
        if provided_col in merged.columns:
            valid |= pd.to_numeric(merged[provided_col], errors="coerce").fillna(0).gt(0)
    return valid & code_match


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

    intraday_prefix = "_intraday_src__"
    auction_prefix = "_auction_src__"
    if intraday_mergeable:
        out = out.merge(_prefixed_source(intraday_prepared, intraday_prefix), on="ts_code", how="left")
    if auction_mergeable:
        out = out.merge(_prefixed_source(auction_prepared, auction_prefix), on="ts_code", how="left")

    intraday_match_col = f"{intraday_prefix}_intraday_code_match"
    auction_match_col = f"{auction_prefix}_auction_code_match"
    intraday_code_match = pd.to_numeric(
        out.get(intraday_match_col, pd.Series(0, index=out.index)), errors="coerce"
    ).fillna(0).gt(0)
    auction_code_match = pd.to_numeric(
        out.get(auction_match_col, pd.Series(0, index=out.index)), errors="coerce"
    ).fillna(0).gt(0)

    intraday_signal_cols = [*INTRADAY_CONTRACT_SCORE_COLS, "open_board_count"]
    intraday_nondegenerate = _nondegenerate_source_features(out, intraday_prefix, intraday_signal_cols, intraday_code_match)
    auction_nondegenerate = _nondegenerate_source_features(out, auction_prefix, AUCTION_CONTRACT_SCORE_COLS, auction_code_match)
    intraday_valid = _row_has_source_feature(out, intraday_prefix, intraday_nondegenerate, intraday_code_match)
    auction_valid = _row_has_source_feature(out, auction_prefix, auction_nondegenerate, auction_code_match)

    if intraday_mergeable:
        materialized: Dict[str, pd.Series] = {}
        for col in intraday_prepared.columns:
            if col == "ts_code":
                continue
            source_col = f"{intraday_prefix}{col}"
            if source_col not in out.columns:
                continue
            if col in out.columns:
                out[col] = out[source_col].combine_first(out[col])
            else:
                materialized[col] = out[source_col]
        if materialized:
            out = pd.concat([out, pd.DataFrame(materialized, index=out.index)], axis=1)

    auction_strength_provided = pd.Series(False, index=out.index, dtype=bool)
    if auction_mergeable:
        auction_provider_cols: Dict[str, pd.Series] = {}
        for col in AUCTION_CONTRACT_SCORE_COLS:
            source_col = f"{auction_prefix}{col}"
            provided_col = f"{auction_prefix}_{col}_provided"
            if source_col not in out.columns or provided_col not in out.columns:
                continue
            provided = pd.to_numeric(out[provided_col], errors="coerce").fillna(0).gt(0) & auction_code_match
            if col in out.columns:
                out.loc[provided, col] = out.loc[provided, source_col]
            else:
                out[col] = out[source_col].where(provided)
            auction_provider_cols[f"_auction_source_{col}_provided"] = provided.astype(int)
            if col == "auction_strength_score":
                auction_strength_provided = provided
        if auction_provider_cols:
            out = pd.concat([out, pd.DataFrame(auction_provider_cols, index=out.index)], axis=1)

    out = ensure_intraday_columns(out, defaults)
    out["intraday_available"] = intraday_valid.astype(int)
    out["auction_available"] = auction_valid.astype(int)
    available = out["intraday_available"]
    auction_available = out["auction_available"]

    if not intraday_present:
        status = pd.Series("missing_file", index=out.index)
        reason = pd.Series("intraday_features.csv missing", index=out.index)
    elif not intraday_mergeable:
        status = pd.Series("missing_columns", index=out.index)
        reason = pd.Series("intraday_features.csv missing ts_code/code column", index=out.index)
    else:
        row_feature_count = pd.to_numeric(
            out.get(f"{intraday_prefix}_intraday_feature_valid_count", 0), errors="coerce"
        ).fillna(0)
        status = pd.Series(
            np.select(
                [~intraday_code_match, row_feature_count <= 0, ~intraday_valid],
                ["missing_stock", "invalid_features", "degenerate_features"],
                default="ok",
            ),
            index=out.index,
        )
        reason = pd.Series(
            np.select(
                [~intraday_code_match, row_feature_count <= 0, ~intraday_valid],
                [
                    "stock not covered by intraday_features.csv",
                    "matched intraday row has no valid 0..1 or 0..100 score feature",
                    "matched intraday row has no non-degenerate feature in candidate intersection",
                ],
                default="",
            ),
            index=out.index,
        )

    out["intraday_status"] = np.where(available > 0, "ok", status)
    out["intraday_data_status"] = out["intraday_status"]
    out["intraday_missing_reason"] = np.where(available > 0, "", reason)
    out["intraday_matched_key"] = np.where(intraday_code_match, out["ts_code"], "")
    out["auction_matched_key"] = np.where(auction_code_match, out["ts_code"], "")

    if not auction_present:
        auction_status = pd.Series("missing_file", index=out.index)
    elif not auction_mergeable:
        auction_status = pd.Series("missing_columns", index=out.index)
    else:
        auction_row_feature_count = pd.to_numeric(
            out.get(f"{auction_prefix}_auction_feature_valid_count", 0), errors="coerce"
        ).fillna(0)
        auction_status = pd.Series(
            np.select(
                [~auction_code_match, auction_row_feature_count <= 0, ~auction_valid],
                ["missing_stock", "invalid_features", "degenerate_features"],
                default="ok",
            ),
            index=out.index,
        )
    out["auction_data_status"] = np.where(auction_available > 0, "ok", auction_status)

    for c in DEFAULT_FLAG_COLS:
        out[c] = pd.to_numeric(out.get(c, 1), errors="coerce").fillna(1).astype(int)
        out.loc[available <= 0, c] = 1
    out.loc[auction_strength_provided, "auction_strength_is_default"] = 0

    out["open_board_risk_score"] = (pd.to_numeric(out["open_board_count"], errors="coerce").fillna(0) / 5.0).clip(0.0, 1.0)
    out["weak_reseal_risk_score"] = (1.0 - pd.to_numeric(out["reseal_score"], errors="coerce").fillna(0.5)).clip(0.0, 1.0)
    auction_strength_has_data = (
        pd.to_numeric(out.get("_auction_strength_score_provided", pd.Series(0, index=out.index)), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(
            out.get("_auction_source_auction_strength_score_provided", pd.Series(0, index=out.index)), errors="coerce"
        ).fillna(0).gt(0)
    )
    auction_volume_has_data = (
        pd.to_numeric(out.get("_auction_real_volume_score_provided", pd.Series(0, index=out.index)), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(
            out.get("_auction_source_auction_real_volume_score_provided", pd.Series(0, index=out.index)), errors="coerce"
        ).fillna(0).gt(0)
    )
    auction_fake = (
        pd.to_numeric(out["auction_strength_score"], errors="coerce").fillna(0.5)
        - pd.to_numeric(out["auction_real_volume_score"], errors="coerce").fillna(0.5)
    ).clip(lower=0.0, upper=1.0)
    out["auction_fake_strength_score"] = auction_fake.where(auction_strength_has_data & auction_volume_has_data, 0.0)
    out["intraday_quality_score"] = calc_intraday_quality_score(out, weights or {})
    out["intraday_soft_risk_score"] = calc_intraday_soft_risk_score(out, soft_risk_weights or {})
    raw_risk = pd.to_numeric(out.get("intraday_risk_score_raw", out.get("intraday_risk_score", 0)), errors="coerce").fillna(0)
    out["intraday_risk_score"] = clip01_series(np.maximum(out["intraday_soft_risk_score"], raw_risk * 0.75))
    out["intraday_confidence_score"] = calc_intraday_confidence_score(out)
    source_cols = [c for c in out.columns if c.startswith(intraday_prefix) or c.startswith(auction_prefix)]
    out = out.drop(columns=source_cols, errors="ignore")
    out.attrs["intraday_nondegenerate_features"] = intraday_nondegenerate
    out.attrs["auction_nondegenerate_features"] = auction_nondegenerate
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
        if c in {*INTRADAY_CONTRACT_SCORE_COLS, *AUCTION_CONTRACT_SCORE_COLS}:
            default_value = _normalize_score_default(defaults.get(c, 0.0), fallback=0.0)
            out[c] = normalize_score_series(out[c]).fillna(default_value)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(float(defaults.get(c, 0.0)))
    for c in [
        "intraday_quality_score",
        "intraday_soft_risk_score",
        "open_board_risk_score",
        "intraday_confidence_score",
        "weak_reseal_risk_score",
        "auction_fake_strength_score",
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


def _contract_distribution(
    df: pd.DataFrame,
    code_match: pd.Series,
    feature_cols: Sequence[str],
    provided_name: Any,
) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    distributions: Dict[str, Dict[str, Any]] = {}
    nondegenerate: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        provided_col = provided_name(col)
        provided = (
            pd.to_numeric(df[provided_col], errors="coerce").fillna(0).gt(0)
            if provided_col in df.columns
            else code_match
        )
        values = pd.to_numeric(df.loc[code_match & provided, col], errors="coerce").dropna()
        if values.empty:
            continue
        unique_count = int(values.nunique(dropna=True))
        if unique_count > 1 or len(values) == 1:
            nondegenerate.append(col)
        bounded_score = col != "open_board_count"
        distributions[col] = {
            "sample_count": int(len(values)),
            "unique_count": unique_count,
            "upper_saturation_rate": float((values >= 1.0 - SCORE_SCALE_EPSILON).mean()) if bounded_score else 0.0,
            "boundary_saturation_rate": float(
                ((values <= SCORE_SCALE_EPSILON) | (values >= 1.0 - SCORE_SCALE_EPSILON)).mean()
            ) if bounded_score else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return distributions, nondegenerate


def build_intraday_debug_summary(full_df: pd.DataFrame, topn_df: pd.DataFrame, intraday_input: Dict[str, Any], trade_date: str) -> Dict[str, Any]:
    full = full_df if isinstance(full_df, pd.DataFrame) else pd.DataFrame()
    topn = topn_df if isinstance(topn_df, pd.DataFrame) else pd.DataFrame()
    avail = pd.to_numeric(full.get("intraday_available", pd.Series([], dtype=float)), errors="coerce").fillna(0)
    top_avail = pd.to_numeric(topn.get("intraday_available", pd.Series([], dtype=float)), errors="coerce").fillna(0)
    status = full.get("intraday_status", pd.Series([], dtype="object")).fillna("").astype(str)
    level = full.get("risk_level", pd.Series([], dtype="object")).fillna("").astype(str)
    label = full.get("risk_label", full.get("risk_tags", pd.Series([], dtype="object"))).fillna("").astype(str)
    intraday_code_match = full.get("intraday_matched_key", pd.Series("", index=full.index)).fillna("").astype(str).ne("")
    auction_code_match = full.get("auction_matched_key", pd.Series("", index=full.index)).fillna("").astype(str).ne("")
    auction_avail = pd.to_numeric(
        full.get("auction_available", pd.Series(0, index=full.index)), errors="coerce"
    ).fillna(0)
    intraday_distributions, intraday_nondegenerate = _contract_distribution(
        full,
        intraday_code_match,
        [*INTRADAY_CONTRACT_SCORE_COLS, "open_board_count"],
        lambda col: f"_{col}_provided",
    )
    auction_distributions, auction_nondegenerate = _contract_distribution(
        full,
        auction_code_match,
        AUCTION_CONTRACT_SCORE_COLS,
        lambda col: f"_auction_source_{col}_provided",
    )
    return {
        "trade_date": str(trade_date),
        "candidate_count": int(len(full)),
        "topn_count": int(len(topn)),
        "intraday_rows": int(intraday_input.get("intraday_rows", intraday_input.get("intraday_features_rows", 0)) or 0),
        "auction_rows": int(intraday_input.get("auction_rows", intraday_input.get("stk_auction_rows", 0)) or 0),
        "auction_source": str(intraday_input.get("auction_source", "")),
        "code_overlap_count": int(intraday_code_match.sum()),
        "code_overlap_rate": float(intraday_code_match.mean()) if len(intraday_code_match) else 0.0,
        "matched_count": int((avail > 0).sum()),
        "matched_rate": float((avail > 0).mean()) if len(avail) else 0.0,
        "valid_feature_count": int((avail > 0).sum()),
        "valid_feature_rate": float((avail > 0).mean()) if len(avail) else 0.0,
        "nondegenerate_features": intraday_nondegenerate,
        "feature_distributions": intraday_distributions,
        "feature_saturation_rates": {
            k: float(v["upper_saturation_rate"]) for k, v in intraday_distributions.items()
        },
        "auction_code_overlap_count": int(auction_code_match.sum()),
        "auction_code_overlap_rate": float(auction_code_match.mean()) if len(auction_code_match) else 0.0,
        "auction_valid_count": int((auction_avail > 0).sum()),
        "auction_valid_rate": float((auction_avail > 0).mean()) if len(auction_avail) else 0.0,
        "auction_nondegenerate_features": auction_nondegenerate,
        "auction_feature_distributions": auction_distributions,
        "auction_feature_saturation_rates": {
            k: float(v["upper_saturation_rate"]) for k, v in auction_distributions.items()
        },
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
