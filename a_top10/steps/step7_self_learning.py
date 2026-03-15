#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step7 : Self Learning — Top10 V3

定位：
- Step7 不再只是“统计命中率 + 顺手训练”
- 而是 V3 的：
    1) 样本成熟判定器
    2) 标签写入器
    3) 质量闸门执行器
    4) 训练放行器
    5) 学习报告产出器

V3 核心输出字段（写回 feature_history.csv）：
- is_sample_mature
- mature_reason
- label_delay_flag
- y_limit_hit
- y_next_ret
- learnable_flag
- reject_reason
- sample_quality_grade
- batch_quality_score
- gate_version

兼容目标：
- 保持 run_step7(s, ctx) -> Dict[str, Any]
- 产出：
    outputs/learning/step7_report_latest.json
    outputs/learning/step7_report_latest.md
    outputs/learning/step7_hit_rate_history.csv
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from a_top10.config import Settings


# ============================================================
# Constants
# ============================================================

VALID_RUN_MODES = {"replay", "train", "auto_daily"}
DEFAULT_TZ = os.getenv("A_TOP10_TZ", "Asia/Shanghai")

GATE_VERSION = "V3_GATE_V1"
LABEL_VERSION = "V3_LABEL_V1"

REQUIRED_HARD_FIELDS = [
    "trade_date",
    "ts_code",
    "StrengthScore",
    "ThemeBoost",
    "Probability",
    "_prob_src",
]

MICROSTRUCTURE_FIELDS = [
    "turnover_rate",
    "open_times",
    "seal_amount",
]

BATCH_KEY_FIELDS = [
    "StrengthScore",
    "ThemeBoost",
    "Probability",
    "turnover_rate",
    "open_times",
    "seal_amount",
]

HIT_HISTORY_COLS = [
    "trade_date",
    "verify_date",
    "topn",
    "hit",
    "hit_rate",
    "note",
]

MIN_LEVEL1_SAMPLES = 20
MIN_LEVEL2_SAMPLES = 50
MIN_LEVEL3_SAMPLES = 100


# ============================================================
# Basic utils
# ============================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _safe_write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_guess(p: Path) -> pd.DataFrame:
    if p is None or not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _now_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_yyyymmdd() -> str:
    try:
        return pd.Timestamp.now(tz=DEFAULT_TZ).strftime("%Y%m%d")
    except Exception:
        return pd.Timestamp.now().strftime("%Y%m%d")


def _safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "<na>", "none"}:
        return ""
    return s


def _normalize_yyyymmdd_value(x: Any) -> str:
    s = _safe_str(x)
    if not s:
        return ""
    if re.match(r"^\d{8}$", s):
        return s
    if re.match(r"^\d{8}\.0+$", s):
        return s.split(".")[0]
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        return digits[:8]
    return s


def _to_numeric_nullable(sr: pd.Series) -> pd.Series:
    return pd.to_numeric(sr, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _to_nosuffix(ts: str) -> str:
    ts = _safe_str(ts)
    if not ts:
        return ""
    return ts.split(".")[0]


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        return d

    code_col = None
    for c in ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]:
        if c in d.columns:
            code_col = c
            break
    if code_col and code_col != "ts_code":
        d["ts_code"] = d[code_col].astype(str)

    if "ts_code" in d.columns:
        d["ts_code"] = d["ts_code"].astype(str).str.strip()

    if "trade_date" in d.columns:
        d["trade_date"] = d["trade_date"].map(_normalize_yyyymmdd_value)
    if "verify_date" in d.columns:
        d["verify_date"] = d["verify_date"].map(_normalize_yyyymmdd_value)

    return d


def _dedup_keep_last_by_keys(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep_keys = [k for k in keys if k in df.columns]
    if not keep_keys:
        return df
    return df.drop_duplicates(subset=keep_keys, keep="last").copy()


def _get_outputs_dir(s: Settings) -> Path:
    try:
        return Path(getattr(s.io, "outputs_dir", "outputs"))
    except Exception:
        return Path("outputs")


def _get_learning_dir(s: Settings) -> Path:
    p = _get_outputs_dir(s) / "learning"
    _ensure_dir(p)
    return p


def _resolve_run_mode() -> str:
    mode = os.getenv("A_TOP10_RUN_MODE", os.getenv("TOP10_RUN_MODE", "auto_daily")).strip().lower()
    if mode not in VALID_RUN_MODES:
        return "auto_daily"
    return mode


# ============================================================
# Warehouse snapshots
# ============================================================

def _warehouse_raw_root_candidates() -> List[Path]:
    return [
        Path("_warehouse/a-share-top3-data/data/raw"),
        Path("outputs/_warehouse/a-share-top3-data/data/raw"),
        Path("data/raw"),
        Path("_warehouse/data/raw"),
    ]


def _warehouse_raw_root() -> Path:
    for cand in _warehouse_raw_root_candidates():
        if cand.exists():
            return cand
    return _warehouse_raw_root_candidates()[0]


def _list_snapshot_dates() -> List[str]:
    base = _warehouse_raw_root()
    if not base.exists():
        return []
    out: List[str] = []
    try:
        for year_dir in base.glob("[0-9][0-9][0-9][0-9]"):
            if not year_dir.is_dir():
                continue
            for ddir in year_dir.iterdir():
                if not ddir.is_dir():
                    continue
                d = ddir.name.strip()
                if re.match(r"^\d{8}$", d):
                    out.append(d)
    except Exception:
        return []
    return sorted(list(dict.fromkeys(out)))


def _latest_snapshot_yyyymmdd() -> str:
    dates = _list_snapshot_dates()
    return dates[-1] if dates else ""


def _upper_bound_yyyymmdd() -> str:
    today = _today_yyyymmdd()
    latest = _latest_snapshot_yyyymmdd()
    if latest and re.match(r"^\d{8}$", latest):
        return min(today, latest)
    return today


def _snapshot_dir(d: str) -> Path:
    return _warehouse_raw_root() / str(d)[:4] / str(d)


def _read_daily_snapshot(d: str) -> pd.DataFrame:
    if not d or not re.match(r"^\d{8}$", str(d)):
        return pd.DataFrame()
    p = _snapshot_dir(d) / "daily.csv"
    df = _read_csv_guess(p)
    return _normalize_id_columns(df)


def _read_limit_list_snapshot(d: str) -> pd.DataFrame:
    if not d or not re.match(r"^\d{8}$", str(d)):
        return pd.DataFrame()
    p = _snapshot_dir(d) / "limit_list_d.csv"
    df = _read_csv_guess(p)
    return _normalize_id_columns(df)


def _next_snapshot_after(trade_date: str, snapshot_dates: List[str], upper_bound: str) -> str:
    td = _safe_str(trade_date)
    ub = _safe_str(upper_bound)
    if not re.match(r"^\d{8}$", td):
        return ""
    if not re.match(r"^\d{8}$", ub):
        return ""
    for d in snapshot_dates:
        if d > td and d <= ub:
            return d
    return ""


# ============================================================
# Feature history
# ============================================================

def _read_feature_history(outputs_dir: Path) -> Tuple[pd.DataFrame, Path]:
    fp = outputs_dir / "learning" / "feature_history.csv"
    if not fp.exists():
        return pd.DataFrame(), fp
    df = _read_csv_guess(fp)
    df = _normalize_id_columns(df)
    return df, fp


def _ensure_v3_feature_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    defaults: Dict[str, Any] = {
        "trade_date": "",
        "ts_code": "",
        "name": "",
        "StrengthScore": np.nan,
        "ThemeBoost": np.nan,
        "Probability": np.nan,
        "_prob_src": "",
        "turnover_rate": np.nan,
        "open_times": np.nan,
        "seal_amount": np.nan,
        "close": np.nan,
        "is_sample_mature": 0,
        "mature_reason": "",
        "label_delay_flag": 0,
        "y_limit_hit": np.nan,
        "y_next_ret": np.nan,
        "learnable_flag": 0,
        "reject_reason": "",
        "sample_quality_grade": "",
        "batch_quality_score": np.nan,
        "gate_version": GATE_VERSION,
        "label_version": LABEL_VERSION,
        "verify_date": "",
    }

    if "probability" in d.columns and "Probability" not in d.columns:
        d["Probability"] = d["probability"]
    if "prob_src" in d.columns and "_prob_src" not in d.columns:
        d["_prob_src"] = d["prob_src"]

    for c, default in defaults.items():
        if c not in d.columns:
            d[c] = default

    d["trade_date"] = d["trade_date"].map(_normalize_yyyymmdd_value)
    d["ts_code"] = d["ts_code"].astype(str).str.strip()
    d["name"] = d["name"].astype(str)

    for c in [
        "StrengthScore",
        "ThemeBoost",
        "Probability",
        "turnover_rate",
        "open_times",
        "seal_amount",
        "close",
        "y_limit_hit",
        "y_next_ret",
        "batch_quality_score",
    ]:
        d[c] = _to_numeric_nullable(d[c])

    for c in ["is_sample_mature", "label_delay_flag", "learnable_flag"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    for c in ["mature_reason", "reject_reason", "sample_quality_grade", "gate_version", "label_version", "_prob_src", "verify_date"]:
        d[c] = d[c].astype(str)
    d["verify_date"] = d["verify_date"].map(_normalize_yyyymmdd_value)

    return d


# ============================================================
# Historical close backfill
# ============================================================

def _build_close_map(daily_df: pd.DataFrame) -> Dict[str, float]:
    if daily_df is None or daily_df.empty:
        return {}
    code_col = None
    for c in ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]:
        if c in daily_df.columns:
            code_col = c
            break
    close_col = None
    for c in ["close", "收盘价"]:
        if c in daily_df.columns:
            close_col = c
            break
    if code_col is None or close_col is None:
        return {}

    out: Dict[str, float] = {}
    for _, row in daily_df.iterrows():
        code = _safe_str(row.get(code_col))
        if not code:
            continue
        close_v = pd.to_numeric(row.get(close_col), errors="coerce")
        if pd.isna(close_v):
            continue
        close_f = float(close_v)
        out[code] = close_f
        out[_to_nosuffix(code)] = close_f
    return out


def _build_trade_date_close_map(trade_date: str) -> Dict[str, float]:
    daily_df = _read_daily_snapshot(trade_date)
    if daily_df is not None and not daily_df.empty:
        close_map = _build_close_map(daily_df)
        if close_map:
            return close_map

    limit_df = _read_limit_list_snapshot(trade_date)
    if limit_df is None or limit_df.empty:
        return {}

    code_col = None
    for c in ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]:
        if c in limit_df.columns:
            code_col = c
            break
    close_col = None
    for c in ["close", "收盘价"]:
        if c in limit_df.columns:
            close_col = c
            break
    if code_col is None or close_col is None:
        return {}

    out: Dict[str, float] = {}
    for _, row in limit_df.iterrows():
        code = _safe_str(row.get(code_col))
        if not code:
            continue
        close_v = pd.to_numeric(row.get(close_col), errors="coerce")
        if pd.isna(close_v):
            continue
        close_f = float(close_v)
        out[code] = close_f
        out[_to_nosuffix(code)] = close_f
    return out


def _backfill_trade_date_close(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()
    if "close" not in d.columns:
        d["close"] = np.nan
    d["close"] = _to_numeric_nullable(d["close"])

    missing_mask = d["close"].isna()
    if not missing_mask.any():
        return d

    for trade_date in sorted(d.loc[missing_mask, "trade_date"].dropna().astype(str).unique().tolist()):
        td = _normalize_yyyymmdd_value(trade_date)
        if not re.match(r"^\d{8}$", td):
            continue
        close_map = _build_trade_date_close_map(td)
        if not close_map:
            warnings.append(f"close_backfill_source_missing: trade_date={td}")
            continue

        mask_td = (d["trade_date"].astype(str) == td) & d["close"].isna()
        if not mask_td.any():
            continue

        codes = d.loc[mask_td, "ts_code"].astype(str)
        filled = codes.map(lambda x: close_map.get(x, close_map.get(_to_nosuffix(x), np.nan)))
        d.loc[mask_td, "close"] = pd.to_numeric(filled, errors="coerce")

    return d


# ============================================================
# Labeling
# ============================================================

def _extract_limit_codes(limit_df: pd.DataFrame) -> set:
    if limit_df is None or limit_df.empty:
        return set()
    code_col = None
    for c in ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]:
        if c in limit_df.columns:
            code_col = c
            break
    if code_col is None:
        return set()

    vals = set()
    for v in limit_df[code_col].astype(str).str.strip().tolist():
        if not v:
            continue
        vals.add(v)
        vals.add(_to_nosuffix(v))
    return vals


def _label_one_day(df_day: pd.DataFrame, verify_date: str, limit_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    d = df_day.copy()
    limit_codes = _extract_limit_codes(limit_df)
    close_map = _build_close_map(daily_df)

    mature_reason = []
    label_delay_flag = []
    is_sample_mature = []
    y_limit_hit = []
    y_next_ret = []

    for _, row in d.iterrows():
        code = _safe_str(row.get("ts_code"))
        if not code:
            is_sample_mature.append(0)
            mature_reason.append("missing_ts_code")
            label_delay_flag.append(1)
            y_limit_hit.append(np.nan)
            y_next_ret.append(np.nan)
            continue

        if limit_df.empty or daily_df.empty:
            is_sample_mature.append(0)
            mature_reason.append("truth_source_missing")
            label_delay_flag.append(1)
            y_limit_hit.append(np.nan)
            y_next_ret.append(np.nan)
            continue

        close_t = pd.to_numeric(row.get("close"), errors="coerce")
        close_next = close_map.get(code, close_map.get(_to_nosuffix(code), np.nan))

        hit = 1.0 if (code in limit_codes or _to_nosuffix(code) in limit_codes) else 0.0
        if np.isnan(close_t) or np.isnan(close_next):
            ret = np.nan
        else:
            try:
                ret = float(close_next) / float(close_t) - 1.0
            except Exception:
                ret = np.nan

        is_sample_mature.append(1)
        mature_reason.append("next_trade_truth_ready")
        label_delay_flag.append(0)
        y_limit_hit.append(hit)
        y_next_ret.append(ret)

    d["verify_date"] = verify_date
    d["is_sample_mature"] = is_sample_mature
    d["mature_reason"] = mature_reason
    d["label_delay_flag"] = label_delay_flag
    d["y_limit_hit"] = y_limit_hit
    d["y_next_ret"] = y_next_ret
    d["label_version"] = LABEL_VERSION
    return d


def _apply_maturity_and_labels(df: pd.DataFrame, snapshot_dates: List[str], upper_bound: str, warnings: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    out_parts: List[pd.DataFrame] = []
    for trade_date, df_day in df.groupby("trade_date", sort=True):
        verify_date = _next_snapshot_after(str(trade_date), snapshot_dates, upper_bound)
        if not verify_date:
            tmp = df_day.copy()
            tmp["verify_date"] = ""
            tmp["is_sample_mature"] = 0
            tmp["mature_reason"] = "next_snapshot_not_ready"
            tmp["label_delay_flag"] = 1
            tmp["y_limit_hit"] = np.nan
            tmp["y_next_ret"] = np.nan
            tmp["label_version"] = LABEL_VERSION
            out_parts.append(tmp)
            continue

        limit_df = _read_limit_list_snapshot(verify_date)
        daily_df = _read_daily_snapshot(verify_date)
        if limit_df.empty or daily_df.empty:
            warnings.append(f"truth_source_not_ready: trade_date={trade_date}, verify_date={verify_date}")
            tmp = df_day.copy()
            tmp["verify_date"] = verify_date
            tmp["is_sample_mature"] = 0
            tmp["mature_reason"] = "truth_source_missing"
            tmp["label_delay_flag"] = 1
            tmp["y_limit_hit"] = np.nan
            tmp["y_next_ret"] = np.nan
            tmp["label_version"] = LABEL_VERSION
            out_parts.append(tmp)
            continue

        out_parts.append(_label_one_day(df_day, verify_date, limit_df, daily_df))

    return pd.concat(out_parts, ignore_index=True) if out_parts else df.copy()


# ============================================================
# Sample gates
# ============================================================

def _sample_reject_reason(row: pd.Series) -> str:
    for c in ["trade_date", "ts_code", "StrengthScore", "ThemeBoost", "Probability", "_prob_src"]:
        v = row.get(c, np.nan)
        if c in {"trade_date", "ts_code", "_prob_src"}:
            if _safe_str(v) == "":
                return f"missing_{c.lower()}"
        else:
            if pd.isna(pd.to_numeric(v, errors="coerce")):
                return f"missing_{c.lower()}"

    if int(pd.to_numeric(row.get("is_sample_mature"), errors="coerce") or 0) != 1:
        return "sample_not_mature"

    if pd.isna(pd.to_numeric(row.get("y_limit_hit"), errors="coerce")):
        return "missing_label"

    micro_missing = 0
    for c in MICROSTRUCTURE_FIELDS:
        if pd.isna(pd.to_numeric(row.get(c), errors="coerce")):
            micro_missing += 1
    if micro_missing == 3:
        return "missing_microstructure_cluster"

    return ""


def _sample_quality_grade(row: pd.Series) -> str:
    hard_ok = True
    for c in REQUIRED_HARD_FIELDS:
        v = row.get(c, np.nan)
        if c in {"trade_date", "ts_code", "_prob_src"}:
            if _safe_str(v) == "":
                hard_ok = False
        else:
            if pd.isna(pd.to_numeric(v, errors="coerce")):
                hard_ok = False

    micro_nonnull = 0
    for c in MICROSTRUCTURE_FIELDS:
        if not pd.isna(pd.to_numeric(row.get(c), errors="coerce")):
            micro_nonnull += 1

    theme_core_nonnull = 0
    for c in ["ThemeBoost", "Probability"]:
        if not pd.isna(pd.to_numeric(row.get(c), errors="coerce")):
            theme_core_nonnull += 1

    if not hard_ok:
        return "D"
    if micro_nonnull >= 2 and theme_core_nonnull >= 2:
        return "A"
    if micro_nonnull >= 1:
        return "B"
    if micro_nonnull == 0:
        return "C"
    return "D"


def _apply_sample_gate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    learnable_flag = []
    reject_reason = []
    quality_grade = []

    for _, row in d.iterrows():
        rr = _sample_reject_reason(row)
        grade = _sample_quality_grade(row)
        quality_grade.append(grade)
        reject_reason.append(rr)
        learnable_flag.append(0 if rr else 1)

    d["learnable_flag"] = learnable_flag
    d["reject_reason"] = reject_reason
    d["sample_quality_grade"] = quality_grade
    d["gate_version"] = GATE_VERSION
    return d


# ============================================================
# Batch gates
# ============================================================

def _batch_quality_score_for_day(df_day: pd.DataFrame) -> float:
    if df_day.empty:
        return 0.0

    score = 0.0
    hard_nonnull = []
    for c in ["trade_date", "ts_code", "StrengthScore", "ThemeBoost", "Probability", "_prob_src", "is_sample_mature", "y_limit_hit"]:
        if c not in df_day.columns:
            hard_nonnull.append(0.0)
            continue
        if c in {"trade_date", "ts_code", "_prob_src"}:
            rate = float((df_day[c].astype(str).str.strip() != "").mean())
        else:
            rate = float(pd.to_numeric(df_day[c], errors="coerce").notna().mean())
        hard_nonnull.append(rate)
    score += 30.0 * float(np.mean(hard_nonnull)) if hard_nonnull else 0.0

    micro_rates = []
    for c in MICROSTRUCTURE_FIELDS:
        if c not in df_day.columns:
            micro_rates.append(0.0)
        else:
            micro_rates.append(float(pd.to_numeric(df_day[c], errors="coerce").notna().mean()))
    score += 25.0 * float(np.mean(micro_rates)) if micro_rates else 0.0

    theme_rates = []
    for c in ["ThemeBoost"]:
        if c not in df_day.columns:
            theme_rates.append(0.0)
        else:
            theme_rates.append(float(pd.to_numeric(df_day[c], errors="coerce").notna().mean()))
    score += 15.0 * float(np.mean(theme_rates)) if theme_rates else 0.0

    label_rates = []
    for c in ["is_sample_mature", "y_limit_hit", "y_next_ret"]:
        if c not in df_day.columns:
            label_rates.append(0.0)
        else:
            label_rates.append(float(pd.to_numeric(df_day[c], errors="coerce").notna().mean()))
    score += 20.0 * float(np.mean(label_rates)) if label_rates else 0.0

    mature_learnable_samples = int(((pd.to_numeric(df_day["is_sample_mature"], errors="coerce").fillna(0) > 0.5) &
                                    (pd.to_numeric(df_day["learnable_flag"], errors="coerce").fillna(0) > 0.5)).sum())
    if mature_learnable_samples >= MIN_LEVEL3_SAMPLES:
        score += 10.0
    elif mature_learnable_samples >= MIN_LEVEL2_SAMPLES:
        score += 7.0
    elif mature_learnable_samples >= MIN_LEVEL1_SAMPLES:
        score += 4.0

    return round(float(score), 4)


def _batch_gate_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    V3 收口：
    - 不再采用“任一 trade_date 失败 => 整批拒训”
    - 改为“坏日期剔除、好日期保留”
    - 只要剩余 pass_dates 上仍有足够成熟可学样本，Step7 就可继续训练
    """
    if df.empty:
        return {
            "pass": False,
            "reason": "feature_history empty",
            "days": {},
            "pass_dates": [],
            "fail_dates": [],
            "eligible_train_rows": 0,
            "eligible_positive_rows": 0,
        }

    report: Dict[str, Any] = {
        "pass": False,
        "reason": "no_pass_trade_dates",
        "days": {},
        "pass_dates": [],
        "fail_dates": [],
        "eligible_train_rows": 0,
        "eligible_positive_rows": 0,
    }

    pass_dates: List[str] = []
    fail_dates: List[str] = []

    for trade_date, df_day in df.groupby("trade_date", sort=True):
        day_rep: Dict[str, Any] = {}

        for c in BATCH_KEY_FIELDS:
            if c not in df_day.columns:
                miss = 1.0
            else:
                miss = float(1.0 - pd.to_numeric(df_day[c], errors="coerce").notna().mean())
            day_rep[f"{c}_missing_rate"] = miss

        mature_learnable_samples = int(((pd.to_numeric(df_day["is_sample_mature"], errors="coerce").fillna(0) > 0.5) &
                                        (pd.to_numeric(df_day["learnable_flag"], errors="coerce").fillna(0) > 0.5)).sum())
        day_rep["mature_learnable_samples"] = mature_learnable_samples
        day_rep["batch_quality_score"] = _batch_quality_score_for_day(df_day)

        fail_reasons: List[str] = []
        if day_rep["StrengthScore_missing_rate"] > 0.30:
            fail_reasons.append("StrengthScore_missing_rate>0.30")
        if day_rep["ThemeBoost_missing_rate"] > 0.30:
            fail_reasons.append("ThemeBoost_missing_rate>0.30")
        if day_rep["Probability_missing_rate"] > 0.20:
            fail_reasons.append("Probability_missing_rate>0.20")
        if day_rep["turnover_rate_missing_rate"] > 0.40:
            fail_reasons.append("turnover_rate_missing_rate>0.40")
        if day_rep["open_times_missing_rate"] > 0.70:
            fail_reasons.append("open_times_missing_rate>0.70")
        if day_rep["seal_amount_missing_rate"] > 0.70:
            fail_reasons.append("seal_amount_missing_rate>0.70")
        if mature_learnable_samples < MIN_LEVEL1_SAMPLES:
            fail_reasons.append(f"mature_learnable_samples<{MIN_LEVEL1_SAMPLES}")
        if float(day_rep["batch_quality_score"]) < 60.0:
            fail_reasons.append("batch_quality_score<60")

        day_pass = len(fail_reasons) == 0
        day_rep["pass"] = bool(day_pass)
        day_rep["fail_reasons"] = fail_reasons
        report["days"][str(trade_date)] = day_rep

        if day_pass:
            pass_dates.append(str(trade_date))
        else:
            fail_dates.append(str(trade_date))

    report["pass_dates"] = pass_dates
    report["fail_dates"] = fail_dates

    eligible_df = df[df["trade_date"].astype(str).isin(pass_dates)].copy() if pass_dates else pd.DataFrame()
    if not eligible_df.empty:
        eligible_train = eligible_df[
            (pd.to_numeric(eligible_df["is_sample_mature"], errors="coerce").fillna(0) > 0.5) &
            (pd.to_numeric(eligible_df["learnable_flag"], errors="coerce").fillna(0) > 0.5) &
            (pd.to_numeric(eligible_df["y_limit_hit"], errors="coerce").notna())
        ].copy()
        report["eligible_train_rows"] = int(len(eligible_train))
        report["eligible_positive_rows"] = int(pd.to_numeric(eligible_train.get("y_limit_hit"), errors="coerce").fillna(0).sum())

    if pass_dates:
        report["pass"] = True
        if fail_dates:
            report["reason"] = "partial_pass_bad_trade_dates_excluded"
        else:
            report["reason"] = "all_trade_dates_passed_batch_gate"
    else:
        report["pass"] = False
        report["reason"] = "no_trade_dates_passed_batch_gate"

    return report


def _write_batch_quality_score(df: pd.DataFrame, batch_gate: Dict[str, Any]) -> pd.DataFrame:
    d = df.copy()
    score_map = {str(td): float(rep.get("batch_quality_score", 0.0)) for td, rep in (batch_gate.get("days", {}) or {}).items()}
    d["batch_quality_score"] = d["trade_date"].astype(str).map(score_map).fillna(0.0)
    return d


# ============================================================
# Hit history
# ============================================================

def _read_hit_history_csv(hit_csv: Path) -> pd.DataFrame:
    if hit_csv is None or not hit_csv.exists():
        return pd.DataFrame(columns=HIT_HISTORY_COLS)
    df = _read_csv_guess(hit_csv)
    if df.empty:
        return pd.DataFrame(columns=HIT_HISTORY_COLS)
    for c in HIT_HISTORY_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[HIT_HISTORY_COLS].copy()


def _build_hit_history(df: pd.DataFrame, learning_dir: Path, warnings: List[str]) -> Tuple[pd.DataFrame, Path, Optional[Dict[str, Any]]]:
    hit_csv = learning_dir / "step7_hit_rate_history.csv"

    if df.empty:
        empty = pd.DataFrame(columns=HIT_HISTORY_COLS)
        empty.to_csv(hit_csv, index=False, encoding="utf-8-sig")
        return empty, hit_csv, None

    rows: List[Dict[str, Any]] = []
    for trade_date, df_day in df.groupby("trade_date", sort=True):
        mature_day = df_day[pd.to_numeric(df_day["is_sample_mature"], errors="coerce").fillna(0) > 0.5].copy()
        pred_day = mature_day.sort_values(["Probability"], ascending=[False]).head(10)

        if pred_day.empty:
            rows.append({
                "trade_date": trade_date,
                "verify_date": "",
                "topn": 0,
                "hit": "",
                "hit_rate": "",
                "note": "pending_or_no_mature_rows",
            })
            continue

        verify_date = _safe_str(pred_day["verify_date"].iloc[0])
        y = pd.to_numeric(pred_day["y_limit_hit"], errors="coerce")
        if y.notna().sum() == 0:
            rows.append({
                "trade_date": trade_date,
                "verify_date": verify_date,
                "topn": len(pred_day),
                "hit": "",
                "hit_rate": "",
                "note": "pending_label",
            })
            continue

        hit = int(y.fillna(0).sum())
        hit_rate = round(float(hit / max(1, len(pred_day))), 4)
        rows.append({
            "trade_date": trade_date,
            "verify_date": verify_date,
            "topn": int(len(pred_day)),
            "hit": int(hit),
            "hit_rate": hit_rate,
            "note": "src=feature_history_v3",
        })

    hit_df = pd.DataFrame(rows, columns=HIT_HISTORY_COLS)
    hit_df.to_csv(hit_csv, index=False, encoding="utf-8-sig")

    latest_hit = None
    done = hit_df[hit_df["hit_rate"].astype(str).str.strip().ne("")]
    if not done.empty:
        latest_hit = done.sort_values("trade_date").iloc[-1].to_dict()

    return hit_df, hit_csv, latest_hit


# ============================================================
# Model training
# ============================================================

def _train_models_pipeline(s: Settings, df: pd.DataFrame, batch_gate: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    run_mode = _resolve_run_mode()
    allow_model_update = False if run_mode == "replay" else True

    out: Dict[str, Any] = {
        "run_mode": run_mode,
        "allow_model_update": allow_model_update,
        "trained": False,
        "updated": False,
        "detail": {},
    }

    pass_dates = [str(x) for x in (batch_gate.get("pass_dates") or [])]
    if pass_dates:
        train_base = df[df["trade_date"].astype(str).isin(pass_dates)].copy()
    else:
        train_base = pd.DataFrame(columns=df.columns)

    mature_train = train_base[
        (pd.to_numeric(train_base.get("is_sample_mature"), errors="coerce").fillna(0) > 0.5) &
        (pd.to_numeric(train_base.get("learnable_flag"), errors="coerce").fillna(0) > 0.5) &
        (pd.to_numeric(train_base.get("y_limit_hit"), errors="coerce").notna())
    ].copy()

    n = int(len(mature_train))
    pos = int(pd.to_numeric(mature_train.get("y_limit_hit"), errors="coerce").fillna(0).sum()) if n else 0

    feature_coverage_vals = []
    for c in ["StrengthScore", "ThemeBoost", "turnover_rate", "seal_amount", "open_times", "Probability"]:
        if c in mature_train.columns and len(mature_train):
            feature_coverage_vals.append(float(pd.to_numeric(mature_train[c], errors="coerce").notna().mean()))
        else:
            feature_coverage_vals.append(0.0)
    feature_coverage = float(np.mean(feature_coverage_vals)) if feature_coverage_vals else 0.0

    level = (
        "below_level1" if n < MIN_LEVEL1_SAMPLES else
        "level1" if n < MIN_LEVEL2_SAMPLES else
        "level2" if n < MIN_LEVEL3_SAMPLES else
        "level3"
    )

    out["detail"] = {
        "train_rows": n,
        "pos": pos,
        "neg": int(n - pos),
        "feature_coverage": feature_coverage,
        "level": level,
        "batch_gate_pass": bool(batch_gate.get("pass")),
        "pass_trade_dates": pass_dates,
        "fail_trade_dates": [str(x) for x in (batch_gate.get("fail_dates") or [])],
        "eligible_train_rows": int(batch_gate.get("eligible_train_rows", 0) or 0),
        "eligible_positive_rows": int(batch_gate.get("eligible_positive_rows", 0) or 0),
    }

    if not pass_dates:
        out["detail"]["reason"] = "skip_train: no_pass_trade_dates"
        warnings.append("skip_train: no_pass_trade_dates")
        return out

    if n < MIN_LEVEL1_SAMPLES:
        out["detail"]["reason"] = "below_level1_min_samples_after_pass_date_filter"
        warnings.append("below_level1_min_samples_after_pass_date_filter")
        return out

    if pos < 12:
        out["detail"]["reason"] = "insufficient_positive_samples_after_pass_date_filter"
        warnings.append("insufficient_positive_samples_after_pass_date_filter")
        return out

    if feature_coverage < 0.85:
        out["detail"]["reason"] = "insufficient_feature_coverage_after_pass_date_filter"
        warnings.append("insufficient_feature_coverage_after_pass_date_filter")
        return out

    try:
        from a_top10.steps.step5_ml_probability import train_step5_models
    except Exception as e:
        out["detail"]["reason"] = f"import_step5_train_failed:{type(e).__name__}"
        warnings.append(out["detail"]["reason"])
        return out

    try:
        train_res = train_step5_models(s=s, lookback=150)
        out["trained"] = bool(train_res.get("ok"))
        out["updated"] = bool(train_res.get("updated"))
        out["detail"]["step5_train_result"] = train_res
        if batch_gate.get("fail_dates"):
            out["detail"]["reason"] = "ok_partial_pass_dates_trained"
        else:
            out["detail"]["reason"] = "ok_all_pass_dates_trained"
        return out
    except Exception as e:
        out["detail"]["reason"] = f"step5_train_failed:{type(e).__name__}"
        warnings.append(out["detail"]["reason"])
        return out


# ============================================================
# Report rendering
# ============================================================

def _md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join([str(r.get(c, "")).strip() for c in cols]) + " |")
    return "\n".join([header, sep] + body)


def render_report_md(report: Dict[str, Any]) -> str:
    latest_hit = report.get("latest_hit")
    train_result = report.get("train_result", {}) or {}
    batch_gate = report.get("batch_gate", {}) or {}
    warnings = report.get("warnings", []) or []
    hit_rows_done_last10 = report.get("hit_rows_done_last10", []) or []

    md_lines: List[str] = []
    md_lines.append("# Step7 自学习报告（latest）")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{report.get('ts','')}")
    md_lines.append(f"- RunMode：{report.get('run_mode','')}")
    md_lines.append(f"- Today：{report.get('today_yyyymmdd','')}")
    md_lines.append(f"- LatestSnapshot：{report.get('latest_snapshot_yyyymmdd','')}")
    md_lines.append(f"- LabelUpperBound：{report.get('label_upper_bound_yyyymmdd','')}")
    md_lines.append("")

    md_lines.append("## 1) 最新命中")
    md_lines.append("")
    if latest_hit:
        md_lines.append(f"- trade_date：{latest_hit.get('trade_date','')}")
        md_lines.append(f"- verify_date：{latest_hit.get('verify_date','')}")
        md_lines.append(f"- hit/topn：{latest_hit.get('hit','')}/{latest_hit.get('topn','')}")
        md_lines.append(f"- hit_rate：{latest_hit.get('hit_rate','')}")
        if latest_hit.get("note"):
            md_lines.append(f"- note：{latest_hit.get('note','')}")
    else:
        md_lines.append("- 暂无可验证命中")

    md_lines.append("")
    md_lines.append("## 1.1) 近10日 Top10 命中率（done-only）")
    md_lines.append("")
    if hit_rows_done_last10:
        md_lines.append(_md_table(hit_rows_done_last10, cols=["trade_date", "verify_date", "topn", "hit", "hit_rate"]))
    else:
        md_lines.append("- 暂无近10日可统计数据")

    md_lines.append("")
    md_lines.append("## 2) 批级闸门")
    md_lines.append("")
    md_lines.append(f"- pass：{batch_gate.get('pass')}")
    md_lines.append(f"- reason：{batch_gate.get('reason','')}")
    md_lines.append(f"- trade_dates：{len(batch_gate.get('days', {}) or {})}")
    md_lines.append(f"- pass_dates：{len(batch_gate.get('pass_dates', []) or [])}")
    md_lines.append(f"- fail_dates：{len(batch_gate.get('fail_dates', []) or [])}")
    md_lines.append(f"- eligible_train_rows：{batch_gate.get('eligible_train_rows', 0)}")

    md_lines.append("")
    md_lines.append("## 3) 训练执行结果")
    md_lines.append("")
    md_lines.append(f"- trained：{train_result.get('trained')}")
    md_lines.append(f"- updated：{train_result.get('updated')}")
    if isinstance(train_result.get("detail"), dict):
        d = train_result["detail"]
        md_lines.append(f"- level：{d.get('level','')}")
        md_lines.append(f"- train_rows：{d.get('train_rows','')}")
        md_lines.append(f"- pos/neg：{d.get('pos','')}/{d.get('neg','')}")
        md_lines.append(f"- feature_coverage：{d.get('feature_coverage','')}")
        md_lines.append(f"- pass_trade_dates：{len(d.get('pass_trade_dates', []) or [])}")
        md_lines.append(f"- fail_trade_dates：{len(d.get('fail_trade_dates', []) or [])}")
        if d.get("reason"):
            md_lines.append(f"- reason：{d.get('reason')}")

    if warnings:
        md_lines.append("")
        md_lines.append("## 4) Warnings")
        md_lines.append("")
        for w in warnings[:80]:
            md_lines.append(f"- {w}")

    return "\n".join(md_lines) + "\n"


# ============================================================
# Main Step7
# ============================================================

def run_step7(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []

    outputs_dir = _get_outputs_dir(s)
    learning_dir = _get_learning_dir(s)

    run_mode = _resolve_run_mode()
    today = _today_yyyymmdd()
    latest_snapshot = _latest_snapshot_yyyymmdd()
    upper_bound = _upper_bound_yyyymmdd()
    snapshot_dates = _list_snapshot_dates()

    fh_raw, fh_path = _read_feature_history(outputs_dir)
    fh_df = _ensure_v3_feature_history_columns(_normalize_id_columns(fh_raw)) if not fh_raw.empty else pd.DataFrame()
    fh_df = _dedup_keep_last_by_keys(fh_df, ["trade_date", "ts_code"])

    if fh_df.empty:
        warnings.append("feature_history empty")
        report = {
            "ts": _now_str(),
            "run_mode": run_mode,
            "today_yyyymmdd": today,
            "latest_snapshot_yyyymmdd": latest_snapshot,
            "label_upper_bound_yyyymmdd": upper_bound,
            "feature_history_file": str(fh_path),
            "batch_gate": {"pass": False, "reason": "feature_history empty", "days": {}, "pass_dates": [], "fail_dates": [], "eligible_train_rows": 0, "eligible_positive_rows": 0},
            "train_result": {"trained": False, "updated": False, "detail": {"reason": "feature_history empty"}},
            "latest_hit": None,
            "hit_rows_done_last10": [],
            "warnings": warnings,
        }
        report_json = learning_dir / "step7_report_latest.json"
        report_md = learning_dir / "step7_report_latest.md"
        _safe_write_json(report_json, report)
        _safe_write_text(report_md, render_report_md(report))
        return {
            "step7_learning": {
                "run_mode": run_mode,
                "feature_history_file": str(fh_path),
                "report_json": str(report_json),
                "report_md": str(report_md),
                "trained": False,
                "updated": False,
                "warnings": warnings,
            }
        }

    fh_df = _backfill_trade_date_close(fh_df, warnings)
    fh_df["close"] = _to_numeric_nullable(fh_df["close"])

    fh_df = _apply_maturity_and_labels(fh_df, snapshot_dates=snapshot_dates, upper_bound=upper_bound, warnings=warnings)
    fh_df = _apply_sample_gate(fh_df)

    batch_gate = _batch_gate_report(fh_df)
    fh_df = _write_batch_quality_score(fh_df, batch_gate)

    fh_df = _dedup_keep_last_by_keys(fh_df, ["trade_date", "ts_code"])
    fh_df.to_csv(fh_path, index=False, encoding="utf-8")

    hit_df, hit_csv, latest_hit = _build_hit_history(fh_df, learning_dir, warnings)

    hit_rows_done_last10: List[Dict[str, Any]] = []
    if hit_df is not None and not hit_df.empty:
        d = hit_df.copy()
        d["hit_rate"] = d["hit_rate"].astype(str).str.strip()
        d = d[d["hit_rate"] != ""].copy()
        if not d.empty:
            d = d.sort_values("trade_date").tail(10)
            hit_rows_done_last10 = d[["trade_date", "verify_date", "topn", "hit", "hit_rate"]].to_dict(orient="records")

    train_result = _train_models_pipeline(s=s, df=fh_df, batch_gate=batch_gate, warnings=warnings)

    report = {
        "ts": _now_str(),
        "run_mode": run_mode,
        "today_yyyymmdd": today,
        "latest_snapshot_yyyymmdd": latest_snapshot,
        "label_upper_bound_yyyymmdd": upper_bound,
        "feature_history_file": str(fh_path),
        "latest_hit": latest_hit,
        "hit_rows_done_last10": hit_rows_done_last10,
        "batch_gate": batch_gate,
        "train_result": train_result,
        "warnings": warnings,
    }

    report_json = learning_dir / "step7_report_latest.json"
    report_md = learning_dir / "step7_report_latest.md"
    _safe_write_json(report_json, report)
    _safe_write_text(report_md, render_report_md(report))

    return {
        "step7_learning": {
            "run_mode": run_mode,
            "feature_history_file": str(fh_path),
            "hit_history_csv": str(hit_csv),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "trained": bool(train_result.get("trained")),
            "updated": bool(train_result.get("updated")),
            "train_rows": int(train_result.get("detail", {}).get("train_rows", 0)) if isinstance(train_result.get("detail"), dict) else 0,
            "level": str(train_result.get("detail", {}).get("level", "")) if isinstance(train_result.get("detail"), dict) else "",
            "warnings": warnings,
        }
    }


if __name__ == "__main__":
    s = Settings()
    out = run_step7(s, ctx={})
    print(json.dumps(out, ensure_ascii=False, indent=2))
