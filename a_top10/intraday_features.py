from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


CODE_COL_CANDIDATES = ["ts_code", "TS_CODE", "code", "CODE", "证券代码", "股票代码", "代码"]

INTRADAY_DEFAULTS: Dict[str, Any] = {
    "intraday_available": 0,
    "auction_available": 0,
    "limitup_quality_score": 0.50,
    "intraday_risk_score": 0.00,
    "late_withdraw_score": 0.00,
    "reseal_score": 0.50,
    "open_board_count": 0,
    "auction_strength_score": 0.50,
    "auction_real_volume_score": 0.50,
    "seal_stability_score": 0.50,
    "intraday_quality_score": 0.50,
    "intraday_risk_penalty": 0.00,
    "intraday_hard_risk_flag": 0,
}

INTRADAY_FEATURE_COLS = [
    "intraday_available",
    "auction_available",
    "limitup_quality_score",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_quality_score",
    "intraday_data_status",
    "auction_data_status",
]

ML_INTRADAY_FEATURES = [
    "limitup_quality_score",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_quality_score",
    "strength_plus_score",
    "intraday_available",
    "auction_available",
]


def get_intraday_cfg(settings: Any) -> Any:
    return getattr(settings, "intraday", None)


def get_intraday_defaults(settings: Any = None) -> Dict[str, Any]:
    cfg = get_intraday_cfg(settings)
    raw = getattr(cfg, "defaults", {}) if cfg is not None else {}
    out = dict(INTRADAY_DEFAULTS)
    if isinstance(raw, dict):
        out.update(raw)
    return out


def get_intraday_dict(settings: Any, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_intraday_cfg(settings)
    raw = getattr(cfg, key, None) if cfg is not None else None
    if isinstance(raw, dict):
        return dict(raw)
    return dict(default or {})


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


def to_float_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series([default] * (0 if df is None else len(df)), dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.fillna(float(default)).astype("float64")


def clip01_series(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype("float64")
    arr = np.asarray(s, dtype=float)
    return pd.Series(np.clip(arr, 0.0, 1.0), dtype="float64")


def _prepare_base(df: pd.DataFrame, defaults: Dict[str, Any], available_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", available_col])
    out = df.copy()
    code_col = first_existing_col(out, CODE_COL_CANDIDATES)
    if code_col is None:
        return pd.DataFrame(columns=["ts_code", available_col])
    out["ts_code"] = out[code_col].map(normalize_ts_code_value)
    out = out[out["ts_code"] != ""].copy()
    out[available_col] = 1
    return out.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)


def _coalesce_score(out: pd.DataFrame, target: str, candidates: Iterable[str], default: float) -> None:
    if target in out.columns:
        out[target] = to_float_series(out, target, default)
        return
    for c in candidates:
        if c in out.columns:
            out[target] = to_float_series(out, c, default)
            return
    out[target] = float(default)


def prepare_intraday_features(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    out = _prepare_base(df, defaults, "intraday_available")
    if out.empty:
        return out
    _coalesce_score(out, "limitup_quality_score", ["limit_up_quality_score", "limitup_quality", "涨停质量"], defaults["limitup_quality_score"])
    _coalesce_score(out, "intraday_risk_score", ["risk_score", "分时风险"], defaults["intraday_risk_score"])
    _coalesce_score(out, "late_withdraw_score", ["tail_withdraw_score", "尾盘撤退", "尾盘风险"], defaults["late_withdraw_score"])
    _coalesce_score(out, "reseal_score", ["reseal_quality_score", "回封分"], defaults["reseal_score"])
    _coalesce_score(out, "open_board_count", ["open_times", "open_board_times", "炸板次数", "开板次数"], defaults["open_board_count"])
    _coalesce_score(out, "seal_stability_score", ["seal_stability", "封单稳定"], defaults["seal_stability_score"])
    _coalesce_score(out, "auction_strength_score", ["auction_score", "竞价强度"], defaults["auction_strength_score"])
    _coalesce_score(out, "auction_real_volume_score", ["auction_volume_score", "竞价真实量"], defaults["auction_real_volume_score"])
    out["open_board_count"] = pd.to_numeric(out["open_board_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for c in [
        "limitup_quality_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "seal_stability_score",
        "auction_strength_score",
        "auction_real_volume_score",
    ]:
        out[c] = clip01_series(out[c])
    return out


def prepare_auction_features(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    out = _prepare_base(df, defaults, "auction_available")
    if out.empty:
        return out
    _coalesce_score(out, "auction_strength_score", ["auction_score", "auction_imbalance_score", "竞价强度"], defaults["auction_strength_score"])
    _coalesce_score(out, "auction_real_volume_score", ["auction_volume_score", "auction_real_volume", "竞价真实量"], defaults["auction_real_volume_score"])
    for c in ["auction_strength_score", "auction_real_volume_score"]:
        out[c] = clip01_series(out[c])
    return out[["ts_code", "auction_available", "auction_strength_score", "auction_real_volume_score"]].drop_duplicates("ts_code", keep="last")


def calc_intraday_quality_score(df: pd.DataFrame, weights: Dict[str, Any]) -> pd.Series:
    w = {
        "limitup_quality": 0.35,
        "reseal_score": 0.25,
        "auction_strength": 0.20,
        "seal_stability": 0.10,
        "late_withdraw_inverse": 0.10,
    }
    if isinstance(weights, dict):
        w.update(weights)
    raw = (
        float(w["limitup_quality"]) * pd.to_numeric(df.get("limitup_quality_score", 0.5), errors="coerce").fillna(0.5)
        + float(w["reseal_score"]) * pd.to_numeric(df.get("reseal_score", 0.5), errors="coerce").fillna(0.5)
        + float(w["auction_strength"]) * pd.to_numeric(df.get("auction_strength_score", 0.5), errors="coerce").fillna(0.5)
        + float(w["seal_stability"]) * pd.to_numeric(df.get("seal_stability_score", 0.5), errors="coerce").fillna(0.5)
        + float(w["late_withdraw_inverse"]) * (1.0 - pd.to_numeric(df.get("late_withdraw_score", 0.0), errors="coerce").fillna(0.0))
    )
    return clip01_series(raw)


def calc_strength_plus_score(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    base_w = float(cfg.get("base_strength_weight", 0.75)) if isinstance(cfg, dict) else 0.75
    intra_w = float(cfg.get("intraday_quality_weight", 0.25)) if isinstance(cfg, dict) else 0.25
    denom = max(1e-12, base_w + intra_w)
    base_w /= denom
    intra_w /= denom
    base = pd.to_numeric(df.get("base_strength_score", df.get("StrengthScore", 0.0)), errors="coerce").fillna(0.0) / 100.0
    quality = pd.to_numeric(df.get("intraday_quality_score", 0.5), errors="coerce").fillna(0.5)
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    plus01 = base.where(available <= 0, base_w * base + intra_w * quality)
    return clip01_series(plus01) * 100.0


def merge_intraday_to_candidates(
    candidates: pd.DataFrame,
    intraday: pd.DataFrame,
    auction: pd.DataFrame,
    defaults: Dict[str, Any],
    weights: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    defaults = {**INTRADAY_DEFAULTS, **(defaults or {})}
    out = candidates.copy()
    if "ts_code" not in out.columns:
        return ensure_intraday_columns(out, defaults)
    out["ts_code"] = out["ts_code"].map(normalize_ts_code_value)

    intraday_prepared = prepare_intraday_features(intraday, defaults)
    auction_prepared = prepare_auction_features(auction, defaults)
    intraday_present = not intraday_prepared.empty
    auction_present = not auction_prepared.empty

    if intraday_present:
        out = out.merge(intraday_prepared, on="ts_code", how="left", suffixes=("", "_intraday"))
    if auction_present:
        out = out.merge(auction_prepared, on="ts_code", how="left", suffixes=("", "_auction"))
        for c in ["auction_strength_score", "auction_real_volume_score"]:
            ac = f"{c}_auction"
            if ac in out.columns:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce").combine_first(pd.to_numeric(out[ac], errors="coerce"))
                else:
                    out[c] = out[ac]
                out = out.drop(columns=[ac], errors="ignore")

    out = ensure_intraday_columns(out, defaults)
    out["intraday_data_status"] = np.where(
        not intraday_present,
        "missing_file",
        np.where(pd.to_numeric(out["intraday_available"], errors="coerce").fillna(0) > 0, "ok", "missing_stock"),
    )
    out["auction_data_status"] = np.where(
        not auction_present,
        "missing_file",
        np.where(pd.to_numeric(out["auction_available"], errors="coerce").fillna(0) > 0, "ok", "missing_stock"),
    )
    out["intraday_quality_score"] = calc_intraday_quality_score(out, weights or {})
    return out


def ensure_intraday_columns(df: pd.DataFrame, defaults: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    defaults = dict(INTRADAY_DEFAULTS if defaults is None else {**INTRADAY_DEFAULTS, **defaults})
    out = df.copy()
    for c, default in defaults.items():
        if c not in out.columns:
            out[c] = default
    for c in [
        "intraday_available",
        "auction_available",
        "limitup_quality_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "open_board_count",
        "auction_strength_score",
        "auction_real_volume_score",
        "seal_stability_score",
        "intraday_quality_score",
        "intraday_risk_penalty",
        "intraday_hard_risk_flag",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(float(defaults.get(c, 0.0)))
    for c in [
        "limitup_quality_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "auction_strength_score",
        "auction_real_volume_score",
        "seal_stability_score",
        "intraday_quality_score",
        "intraday_risk_penalty",
    ]:
        out[c] = clip01_series(out[c])
    out["open_board_count"] = out["open_board_count"].clip(lower=0.0)
    if "intraday_data_status" not in out.columns:
        out["intraday_data_status"] = np.where(out["intraday_available"] > 0, "ok", "missing_file")
    if "auction_data_status" not in out.columns:
        out["auction_data_status"] = np.where(out["auction_available"] > 0, "ok", "missing_file")
    return out


def calc_intraday_risk_penalty(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    risk_w = float(cfg.get("risk_penalty_weight", 0.05)) if isinstance(cfg, dict) else 0.05
    late_w = float(cfg.get("late_withdraw_penalty_weight", 0.04)) if isinstance(cfg, dict) else 0.04
    open_w = float(cfg.get("open_board_penalty_weight", 0.02)) if isinstance(cfg, dict) else 0.02
    risk = pd.to_numeric(df.get("intraday_risk_score", 0.0), errors="coerce").fillna(0.0)
    late = pd.to_numeric(df.get("late_withdraw_score", 0.0), errors="coerce").fillna(0.0)
    open_board = pd.to_numeric(df.get("open_board_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    return (risk_w * risk + late_w * late + open_w * (open_board / 5.0).clip(0.0, 1.0)).astype("float64")


def calc_intraday_bonus(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    q_w = float(cfg.get("quality_bonus_weight", 0.03)) if isinstance(cfg, dict) else 0.03
    a_w = float(cfg.get("auction_bonus_weight", 0.015)) if isinstance(cfg, dict) else 0.015
    quality = pd.to_numeric(df.get("intraday_quality_score", 0.5), errors="coerce").fillna(0.5)
    auction = pd.to_numeric(df.get("auction_strength_score", 0.5), errors="coerce").fillna(0.5)
    available = pd.to_numeric(df.get("intraday_available", 0), errors="coerce").fillna(0)
    auction_available = pd.to_numeric(df.get("auction_available", 0), errors="coerce").fillna(0)
    return (q_w * quality * (available > 0).astype(float) + a_w * auction * (auction_available > 0).astype(float)).astype("float64")


def calc_intraday_hard_flags(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    hard = dict(cfg or {})
    enabled = bool(hard.get("enable", True))
    out = df.copy()
    if not enabled:
        out["intraday_hard_risk_flag"] = 0
        return out
    late_max = float(hard.get("max_late_withdraw_score", 0.80))
    risk_max = float(hard.get("max_intraday_risk_score", 0.85))
    open_max = float(hard.get("max_open_board_count", 4))
    quality_min = float(hard.get("min_limitup_quality_score", 0.30))
    flag = (
        (pd.to_numeric(out.get("late_withdraw_score", 0), errors="coerce").fillna(0) >= late_max)
        | (pd.to_numeric(out.get("intraday_risk_score", 0), errors="coerce").fillna(0) >= risk_max)
        | (pd.to_numeric(out.get("open_board_count", 0), errors="coerce").fillna(0) >= open_max)
        | (pd.to_numeric(out.get("limitup_quality_score", 0.5), errors="coerce").fillna(0.5) <= quality_min)
    )
    out["intraday_hard_risk_flag"] = flag.astype(int)
    return out


def build_risk_tags(row: pd.Series) -> str:
    tags = []
    if float(row.get("late_withdraw_score", 0) or 0) >= 0.70:
        tags.append("尾盘撤退")
    if float(row.get("intraday_risk_score", 0) or 0) >= 0.75:
        tags.append("分时高风险")
    if float(row.get("open_board_count", 0) or 0) >= 3:
        tags.append("多次炸板")
    if float(row.get("reseal_score", 0.5) or 0.5) <= 0.35:
        tags.append("回封弱")
    if float(row.get("limitup_quality_score", 0.5) or 0.5) <= 0.40:
        tags.append("涨停质量弱")
    if float(row.get("auction_strength_score", 0.5) or 0.5) <= 0.35:
        tags.append("竞价弱")
    if float(row.get("intraday_available", 0) or 0) <= 0:
        tags.append("分时缺失")
    return "|".join(tags)


def build_intraday_conclusion(row: pd.Series) -> str:
    tags = build_risk_tags(row)
    if float(row.get("intraday_available", 0) or 0) <= 0:
        return "分时数据缺失，已按中性值降级"
    if tags:
        return f"分时风险：{tags}"
    return "分时结构健康"
