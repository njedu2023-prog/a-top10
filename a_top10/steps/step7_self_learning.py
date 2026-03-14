#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


# ============================================================
# Constants
# ============================================================
LR_MODEL_PATH = Path("step5_lr.joblib")
LGBM_MODEL_PATH = Path("step5_lgbm.joblib")

MIN_SAMPLES = 200
MIN_POS = 10

DEFAULT_TZ = os.getenv("A_TOP10_TZ", "Asia/Shanghai")
SAMPLING_STATE_FILE = "sampling_state.json"

V2_REQUIRED_PRED_COLS = [
    "trade_date",
    "verify_date",
    "rank",
    "ts_code",
    "name",
    "prob_rule",
    "prob_ml",
    "prob_final",
    "prob",
    "Probability",
    "final_score",
    "score",
    "StrengthScore",
    "ThemeBoost",
    "board",
    "run_id",
    "run_attempt",
    "commit_sha",
    "generated_at_utc",
]

CORE_FEATURE_COLS_V2 = [
    "StrengthScore",
    "ThemeBoost",
    "turnover_rate",
    "seal_amount",
    "open_times",
    "prob_final",
    "final_score",
]
META_COLS_V2 = ["trade_date", "ts_code", "_prob_src"]

HIT_HISTORY_COLS = [
    "trade_date",
    "expected_next_trade_date",
    "actual_next_trade_date",
    "topn",
    "hit",
    "hit_rate",
    "note",
]


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
    if x is None:
        return ""
    return str(x).strip()


def _to_nosuffix(ts: str) -> str:
    ts = str(ts).strip()
    if not ts:
        return ts
    return ts.split(".")[0]


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _get_outputs_dir(s: Settings) -> Path:
    try:
        return Path(getattr(s.io, "outputs_dir", "outputs"))
    except Exception:
        return Path("outputs")


def _get_models_dir(s: Settings) -> Path:
    try:
        if hasattr(s, "data_repo") and hasattr(s.data_repo, "models_dir"):
            p = Path(getattr(s.data_repo, "models_dir"))
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass
    p = Path("models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_csv_guess(p: Path) -> pd.DataFrame:
    if p is None or not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc)
        except Exception:
            continue
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        return pd.DataFrame()


# ============================================================
# Warehouse snapshots only (V2 only)
# ============================================================
def _warehouse_raw_root() -> Path:
    return Path("_warehouse/a-share-top3-data/data/raw")


def _latest_snapshot_yyyymmdd() -> str:
    base = _warehouse_raw_root()
    if not base.exists():
        return ""
    best = ""
    try:
        for year_dir in base.glob("[0-9][0-9][0-9][0-9]"):
            if not year_dir.is_dir():
                continue
            for ddir in year_dir.iterdir():
                if not ddir.is_dir():
                    continue
                d = ddir.name.strip()
                if re.match(r"^\d{8}$", d) and d > best:
                    best = d
    except Exception:
        return ""
    return best


def _upper_bound_yyyymmdd() -> str:
    today = _today_yyyymmdd()
    latest = _latest_snapshot_yyyymmdd()
    if latest and re.match(r"^\d{8}$", latest):
        return min(today, latest)
    return today


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


def _read_limit_list_warehouse(d: str, warnings: List[str]) -> pd.DataFrame:
    if not d or not re.match(r"^\d{8}$", str(d)):
        return pd.DataFrame()

    y = str(d)[:4]
    p = _warehouse_raw_root() / y / str(d) / "limit_list_d.csv"
    if not p.exists():
        warnings.append(f"limit_list_d missing: {p}")
        return pd.DataFrame()

    df = _read_csv_guess(p)
    if df.empty:
        warnings.append(f"limit_list_d empty: {p}")
    return df


def _limit_codes_from_df(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []

    code_col = None
    for c in ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]:
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        return []

    vals: List[str] = []
    for v in df[code_col].astype(str).str.strip().tolist():
        if not v:
            continue
        vals.append(v)
        vals.append(_to_nosuffix(v))
    return vals


def _next_snapshot_after(trade_date: str, snapshot_dates: List[str], upper_bound: str) -> str:
    td = str(trade_date).strip()
    ub = str(upper_bound).strip()
    if not re.match(r"^\d{8}$", td):
        return ""
    if not re.match(r"^\d{8}$", ub):
        return ""
    for d in snapshot_dates:
        if d > td and d <= ub:
            return d
    return ""


# ============================================================
# V2 pred history only
# ============================================================
def _read_pred_top10_history(outputs_dir: Path, warnings: List[str]) -> pd.DataFrame:
    fp = outputs_dir / "learning" / "pred_top10_history.csv"
    if not fp.exists():
        warnings.append(f"pred_top10_history missing: {fp}")
        return pd.DataFrame()

    df = _read_csv_guess(fp)
    if df.empty:
        warnings.append("pred_top10_history empty")
        return pd.DataFrame()

    missing = [c for c in ["trade_date", "verify_date", "rank", "ts_code", "prob_final", "final_score"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"V2 contract violated: pred_top10_history missing required cols: {missing}")

    return df


def _pred_dates_from_history(pred_df: pd.DataFrame) -> List[str]:
    if pred_df is None or pred_df.empty:
        return []
    dates = pred_df["trade_date"].astype(str).str.strip()
    dates = [d for d in dates.unique().tolist() if re.match(r"^\d{8}$", d)]
    return sorted(dates)


def _canonicalize_pred_history_v2(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(columns=V2_REQUIRED_PRED_COLS)

    d = pred_df.copy()
    d["trade_date"] = d["trade_date"].astype(str).str.strip()
    d["verify_date"] = d["verify_date"].astype(str).str.strip()
    d["ts_code"] = d["ts_code"].astype(str).str.strip()

    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    d["prob_final"] = pd.to_numeric(d["prob_final"], errors="coerce")
    d["final_score"] = pd.to_numeric(d["final_score"], errors="coerce")

    if "StrengthScore" in d.columns:
        d["StrengthScore"] = pd.to_numeric(d["StrengthScore"], errors="coerce")
    else:
        d["StrengthScore"] = pd.NA

    if "ThemeBoost" in d.columns:
        d["ThemeBoost"] = pd.to_numeric(d["ThemeBoost"], errors="coerce")
    else:
        d["ThemeBoost"] = pd.NA

    if "prob_rule" in d.columns:
        d["prob_rule"] = pd.to_numeric(d["prob_rule"], errors="coerce")
    else:
        d["prob_rule"] = pd.NA

    if "prob_ml" in d.columns:
        d["prob_ml"] = pd.to_numeric(d["prob_ml"], errors="coerce")
    else:
        d["prob_ml"] = pd.NA

    for c in V2_REQUIRED_PRED_COLS:
        if c not in d.columns:
            d[c] = ""

    d = d[V2_REQUIRED_PRED_COLS].copy()
    return d


def _get_pred_codes_for_date(pred_df: pd.DataFrame, trade_date: str, topn: int) -> Tuple[List[str], str]:
    if pred_df is None or pred_df.empty:
        return [], "pred_df_empty"

    d = pred_df[pred_df["trade_date"] == str(trade_date)].copy()
    if d.empty:
        return [], "pred_no_rows_for_date"

    d = d.sort_values(["rank", "prob_final", "final_score"], ascending=[True, False, False], kind="mergesort")
    codes = d["ts_code"].astype(str).str.strip().tolist()
    codes = _dedup_keep_order(codes)[: int(topn)]
    return codes, "ok"


def _get_verify_date_from_pred(pred_df: pd.DataFrame, trade_date: str) -> str:
    if pred_df is None or pred_df.empty:
        return ""
    d = pred_df[pred_df["trade_date"] == str(trade_date)].copy()
    if d.empty:
        return ""
    v = d["verify_date"].astype(str).str.strip()
    v = v[(v != "") & (v.str.lower() != "nan")]
    if len(v) == 0:
        return ""
    try:
        return v.value_counts().index[0]
    except Exception:
        return v.iloc[0]


# ============================================================
# Feature history V2 only
# ============================================================
def _read_feature_history(outputs_dir: Path) -> Tuple[pd.DataFrame, str]:
    fp = outputs_dir / "learning" / "feature_history.csv"
    if not fp.exists():
        return pd.DataFrame(), str(fp)
    return _read_csv_guess(fp), str(fp)


def _normalize_feature_history_v2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    required = ["trade_date", "ts_code"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"V2 contract violated: feature_history missing required cols: {missing}")

    for c in CORE_FEATURE_COLS_V2:
        if c not in d.columns:
            d[c] = "0"

    if "_prob_src" not in d.columns:
        d["_prob_src"] = ""

    d["trade_date"] = d["trade_date"].astype(str).str.strip()
    d["ts_code"] = d["ts_code"].astype(str).str.strip()

    for c in CORE_FEATURE_COLS_V2:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


def _rows_per_day_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "trade_date" not in df.columns:
        return pd.Series(dtype="int64")
    d = df.copy()
    d["trade_date"] = d["trade_date"].astype(str).str.strip()
    if "ts_code" not in d.columns:
        d["ts_code"] = ""
    return d.groupby("trade_date")["ts_code"].count().sort_index()


def _quality_gate_v2(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {
        "core_cols": CORE_FEATURE_COLS_V2,
        "meta_cols": META_COLS_V2,
    }

    if df is None or df.empty:
        details["reason"] = "feature_history empty"
        details["pass"] = False
        return False, details

    d = _normalize_feature_history_v2(df)

    missing = [c for c in (META_COLS_V2 + CORE_FEATURE_COLS_V2) if c not in d.columns]
    details["missing_cols"] = missing
    if missing:
        details["reason"] = "missing required cols"
        details["pass"] = False
        return False, details

    dates = sorted(d["trade_date"].astype(str).unique())
    recent_dates = dates[-30:] if len(dates) >= 30 else dates
    w = d[d["trade_date"].astype(str).isin(recent_dates)].copy()

    if w.empty:
        details["reason"] = "window empty"
        details["pass"] = False
        return False, details

    metrics: Dict[str, Any] = {}
    for c in CORE_FEATURE_COLS_V2:
        x = pd.to_numeric(w[c], errors="coerce")
        x0 = x.fillna(0.0)
        metrics[c] = {
            "non_null_rate": float(x.notna().mean()) if len(x) else 0.0,
            "non_zero_rate": float((x0 != 0.0).mean()) if len(x0) else 0.0,
            "std": float(x0.std()) if len(x0) else 0.0,
            "unique_count": int(x.dropna().nunique()),
        }

    try:
        src = w["_prob_src"].astype(str).str.lower().fillna("")
        pseudo_ratio = float((src == "pseudo").mean())
    except Exception:
        pseudo_ratio = 1.0

    details["metrics"] = metrics
    details["pseudo_ratio"] = pseudo_ratio
    details["window_days"] = int(len(sorted(w["trade_date"].astype(str).unique())))
    details["rows_in_window"] = int(len(w))

    ok = True
    for c in CORE_FEATURE_COLS_V2:
        if metrics[c]["non_null_rate"] < 0.90:
            ok = False

    for c in ["StrengthScore", "ThemeBoost", "prob_final", "final_score"]:
        if metrics.get(c, {}).get("std", 0.0) <= 0.0:
            ok = False

    if metrics.get("StrengthScore", {}).get("non_zero_rate", 0.0) < 0.30:
        ok = False
    if metrics.get("turnover_rate", {}).get("non_zero_rate", 0.0) < 0.30:
        ok = False
    if metrics.get("ThemeBoost", {}).get("unique_count", 0) < 6:
        ok = False
    if metrics.get("seal_amount", {}).get("non_zero_rate", 0.0) < 0.05:
        ok = False
    if (metrics.get("open_times", {}).get("std", 0.0) <= 0.0) and (metrics.get("open_times", {}).get("non_zero_rate", 0.0) < 0.02):
        ok = False

    details["pass"] = bool(ok)
    return bool(ok), details


def _count_ge(window: List[int], th: int) -> int:
    return int(sum(1 for v in window if v >= th))


def _decide_sampling_stage(prev_stage: str, days_covered: int, rows_last: List[int], quality_pass: bool) -> Tuple[str, int, Dict[str, Any]]:
    prev = (prev_stage or "S1_MVP").strip()
    rows = [int(x) for x in (rows_last or []) if x is not None]
    debug: Dict[str, Any] = {"prev_stage": prev, "days_covered": int(days_covered), "rows_last": rows[-20:]}

    stage = prev if prev else "S1_MVP"
    target = 200 if stage.startswith("S1") else 500 if stage.startswith("S2") else 1000

    if not quality_pass:
        debug["reason"] = "quality_gate_fail"
        return stage, target, debug

    if stage.startswith("S1"):
        w = rows[-10:]
        if days_covered >= 30 and _count_ge(w, 200) >= 7:
            return "S2_STD", 500, {**debug, "upgrade": "S1->S2"}

    if stage.startswith("S2"):
        w = rows[-15:]
        if days_covered >= 90 and _count_ge(w, 500) >= 10:
            return "S3_STRONG", 1000, {**debug, "upgrade": "S2->S3"}

    if stage.startswith("S3"):
        w = rows[-20:]
        debug["keep_check_ge_1000"] = _count_ge(w, 1000)
        return "S3_STRONG", 1000, debug

    return stage, target, debug


def _write_sampling_state(outputs_dir: Path, obj: Dict[str, Any]) -> str:
    p = outputs_dir / "learning" / SAMPLING_STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# ============================================================
# Hit history freeze-upsert
# ============================================================
def _read_hit_history_csv(hit_csv: Path) -> pd.DataFrame:
    if hit_csv is None or (not hit_csv.exists()):
        return pd.DataFrame(columns=HIT_HISTORY_COLS)
    df = _read_csv_guess(hit_csv)
    if df.empty:
        return pd.DataFrame(columns=HIT_HISTORY_COLS)
    for c in HIT_HISTORY_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[HIT_HISTORY_COLS].copy()


def _is_done_df_row(df: pd.DataFrame, idx: int) -> bool:
    try:
        a = str(df.at[idx, "actual_next_trade_date"]).strip() if "actual_next_trade_date" in df.columns else ""
        hr = str(df.at[idx, "hit_rate"]).strip() if "hit_rate" in df.columns else ""
        return bool(a) and bool(hr) and hr.lower() != "nan"
    except Exception:
        return False


def _freeze_upsert_hit_history(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_old is None or df_old.empty:
        return df_new.copy() if df_new is not None else pd.DataFrame(columns=HIT_HISTORY_COLS)

    d_old = df_old.copy()
    d_new = df_new.copy() if df_new is not None else pd.DataFrame(columns=HIT_HISTORY_COLS)

    for c in HIT_HISTORY_COLS:
        if c not in d_old.columns:
            d_old[c] = ""
        if c not in d_new.columns:
            d_new[c] = ""

    d_old["trade_date"] = d_old["trade_date"].astype(str).str.strip()
    d_new["trade_date"] = d_new["trade_date"].astype(str).str.strip()

    old_map: Dict[str, int] = {}
    for i in range(len(d_old)):
        td = str(d_old.iloc[i].get("trade_date", "")).strip()
        if td and re.match(r"^\d{8}$", td) and td not in old_map:
            old_map[td] = i

    rows_out: List[Dict[str, Any]] = []
    for i in range(len(d_old)):
        rows_out.append({c: str(d_old.iloc[i].get(c, "")).strip() for c in HIT_HISTORY_COLS})

    for j in range(len(d_new)):
        td = str(d_new.iloc[j].get("trade_date", "")).strip()
        if not td or not re.match(r"^\d{8}$", td):
            continue

        new_row = {c: str(d_new.iloc[j].get(c, "")).strip() for c in HIT_HISTORY_COLS}

        if td not in old_map:
            rows_out.append(new_row)
            continue

        i_old = old_map[td]
        old_done = _is_done_df_row(d_old, i_old)
        if old_done:
            continue

        rows_out[i_old] = new_row

    out = pd.DataFrame(rows_out)
    if not out.empty and "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype(str).str.strip()
        out = out.drop_duplicates(subset=["trade_date"], keep="first")
    return out


def _write_hit_history_with_pending_first(hit_csv: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        pd.DataFrame(columns=HIT_HISTORY_COLS).to_csv(hit_csv, index=False, encoding="utf-8-sig")
        return

    d = df.copy()
    for c in HIT_HISTORY_COLS:
        if c not in d.columns:
            d[c] = ""

    d["trade_date"] = d["trade_date"].astype(str).str.strip()

    actual = d["actual_next_trade_date"].astype(str).str.strip()
    hr = d["hit_rate"].astype(str).str.strip()
    pending_mask = (actual == "") | (hr == "") | (hr.str.lower() == "nan")

    df_pending = d[pending_mask].copy()
    df_done = d[~pending_mask].copy()

    if not df_pending.empty:
        df_pending = df_pending.sort_values("trade_date")
    if not df_done.empty:
        df_done = df_done.sort_values("trade_date")

    out = pd.concat([df_pending, df_done], ignore_index=True)
    out = out[HIT_HISTORY_COLS].copy()
    out.to_csv(hit_csv, index=False, encoding="utf-8-sig")


# ============================================================
# Build hit history (V2 only)
# ============================================================
def build_hit_history(
    outputs_dir: Path,
    learning_dir: Path,
    topn: int,
    upper_bound: str,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Path, Optional[Dict[str, Any]]]:
    hit_rows_new: List[Dict[str, Any]] = []
    snapshot_dates = _list_snapshot_dates()

    pred_df_raw = _read_pred_top10_history(outputs_dir, warnings)
    pred_df = _canonicalize_pred_history_v2(pred_df_raw)

    if pred_df.empty:
        raise RuntimeError("V2 contract violated: pred_top10_history.csv unavailable or empty")

    pred_dates = _pred_dates_from_history(pred_df)

    for td in pred_dates:
        codes, reason = _get_pred_codes_for_date(pred_df, td, topn=topn)
        if not codes:
            hit_rows_new.append({
                "trade_date": td,
                "expected_next_trade_date": "",
                "actual_next_trade_date": "",
                "topn": 0,
                "hit": "",
                "hit_rate": "",
                "note": f"pred_table_empty: {reason}",
            })
            continue

        expected_nd = _get_verify_date_from_pred(pred_df, td)
        if not expected_nd:
            expected_nd = _next_snapshot_after(td, snapshot_dates, upper_bound)

        expected_nd = _safe_str(expected_nd)
        if expected_nd.lower() == "nan":
            expected_nd = ""

        if not expected_nd:
            hit_rows_new.append({
                "trade_date": td,
                "expected_next_trade_date": "",
                "actual_next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": "pending_label: no next snapshot available yet",
            })
            continue

        if str(expected_nd) > str(upper_bound):
            hit_rows_new.append({
                "trade_date": td,
                "expected_next_trade_date": expected_nd,
                "actual_next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": f"pending_label: expected_next_trade_date_not_available (upper_bound={upper_bound})",
            })
            continue

        lim_df = _read_limit_list_warehouse(expected_nd, warnings)
        if lim_df is None or lim_df.empty:
            hit_rows_new.append({
                "trade_date": td,
                "expected_next_trade_date": expected_nd,
                "actual_next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": "pending_label: limit_list_d not ready",
            })
            continue

        lim_set = set(_limit_codes_from_df(lim_df))
        hit = 0
        for c in codes:
            if (c in lim_set) or (_to_nosuffix(c) in lim_set):
                hit += 1
        hit_rate = hit / max(1, len(codes))

        hit_rows_new.append({
            "trade_date": td,
            "expected_next_trade_date": expected_nd,
            "actual_next_trade_date": expected_nd,
            "topn": len(codes),
            "hit": int(hit),
            "hit_rate": round(float(hit_rate), 4),
            "note": "src=pred_top10_history_v2",
        })

    hit_csv = learning_dir / "step7_hit_rate_history.csv"
    df_old = _read_hit_history_csv(hit_csv)
    df_new = pd.DataFrame(hit_rows_new) if hit_rows_new else pd.DataFrame(columns=HIT_HISTORY_COLS)

    df_merged = _freeze_upsert_hit_history(df_old, df_new)
    _write_hit_history_with_pending_first(hit_csv, df_merged)

    latest_hit: Optional[Dict[str, Any]] = None
    if df_merged is not None and not df_merged.empty:
        d = df_merged.copy()
        d["trade_date"] = d["trade_date"].astype(str).str.strip()
        a = d["actual_next_trade_date"].astype(str).str.strip()
        hr = d["hit_rate"].astype(str).str.strip()
        done = d[(a != "") & (hr != "") & (hr.str.lower() != "nan")].copy()
        if not done.empty:
            done = done.sort_values("trade_date")
            latest_hit = done.iloc[-1].to_dict()

    return df_merged, hit_csv, latest_hit


# ============================================================
# Build V2 train set from feature_history
# ============================================================
def _build_train_set_from_feature_history(
    outputs_dir: Path,
    lookback_days: int,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    fp = outputs_dir / "learning" / "feature_history.csv"

    meta: Dict[str, Any] = {
        "feature_history_file": str(fp),
        "rows_raw": 0,
        "rows_used": 0,
        "rows_dropped_allzero": 0,
        "dates_total": 0,
        "dates_used": 0,
    }

    if not fp.exists():
        warnings.append("feature_history.csv not found: training skipped.")
        return pd.DataFrame(), meta

    raw = _read_csv_guess(fp)
    if raw.empty:
        warnings.append("feature_history.csv empty: training skipped.")
        return pd.DataFrame(), meta

    df = _normalize_feature_history_v2(raw)
    meta["rows_raw"] = int(len(df))

    dates = sorted([d for d in df["trade_date"].unique().tolist() if re.match(r"^\d{8}$", str(d))])
    meta["dates_total"] = int(len(dates))
    use_dates = dates[-lookback_days:] if len(dates) > lookback_days else dates
    meta["dates_used"] = int(len(use_dates))

    dfx = df[df["trade_date"].isin(use_dates)].copy()
    if dfx.empty:
        warnings.append("feature_history filtered empty by lookback.")
        return pd.DataFrame(), meta

    feats = CORE_FEATURE_COLS_V2
    for c in feats:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce").fillna(0.0)

    allzero = (dfx[feats].abs().sum(axis=1) <= 0.0)
    meta["rows_dropped_allzero"] = int(allzero.sum())
    dfx = dfx[~allzero].copy()
    if dfx.empty:
        warnings.append("all rows are all-zero features: training skipped.")
        return pd.DataFrame(), meta

    snapshot_dates = _list_snapshot_dates()
    upper_bound = _upper_bound_yyyymmdd()

    rows: List[Dict[str, Any]] = []
    for d in sorted([x for x in dfx["trade_date"].unique().tolist() if re.match(r"^\d{8}$", str(x))]):
        verify_d = _next_snapshot_after(d, snapshot_dates, upper_bound)
        if not verify_d:
            warnings.append(f"label_pending: trade_date={d} no next snapshot")
            continue

        lim_df = _read_limit_list_warehouse(verify_d, warnings)
        if lim_df.empty:
            warnings.append(f"label_source_empty: trade_date={d} verify_date={verify_d}")
            continue

        lim_set = set(_limit_codes_from_df(lim_df))
        df_day = dfx[dfx["trade_date"] == d].copy()
        if df_day.empty:
            continue

        for _, r in df_day.iterrows():
            code = str(r.get("ts_code", "")).strip()
            y = 1 if (code in lim_set or _to_nosuffix(code) in lim_set) else 0

            row: Dict[str, Any] = {
                "trade_date": d,
                "verify_date": verify_d,
                "ts_code": code,
                "label": int(y),
            }
            for c in feats:
                row[c] = float(r.get(c, 0.0))
            rows.append(row)

    train_df = pd.DataFrame(rows)
    meta["rows_used"] = int(len(train_df))
    return train_df, meta


# ============================================================
# Train models and save
# ============================================================
def _train_lr(train_df: pd.DataFrame, warnings: List[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats = CORE_FEATURE_COLS_V2
    X = train_df[feats].astype(float).values
    y = train_df["label"].astype(int).values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model


def _train_lgbm(train_df: pd.DataFrame, warnings: List[str]):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        warnings.append("lightgbm not installed: skip lgbm.")
        return None

    feats = CORE_FEATURE_COLS_V2
    X = train_df[feats].astype(float).values
    y = train_df["label"].astype(int).values

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _save_joblib_model(model, path: Path, warnings: List[str]) -> bool:
    if model is None:
        return False
    try:
        import joblib
    except Exception:
        warnings.append("joblib not installed: cannot save model.")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return True
    except Exception as e:
        warnings.append(f"save model failed: {e}")
        return False


def _train_and_save_models(s: Settings, train_df: pd.DataFrame, warnings: List[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"trained": False, "lr_saved": False, "lgbm_saved": False, "models_dir": "", "detail": {}}

    if train_df is None or train_df.empty:
        warnings.append("train_df empty: training skipped.")
        res["detail"] = {"reason": "train_df empty", "train_rows": 0, "pos": 0, "neg": 0}
        return res

    pos = int(train_df["label"].astype(int).sum())
    neg = int(len(train_df) - pos)
    n = int(len(train_df))

    res["detail"]["train_rows"] = n
    res["detail"]["pos"] = pos
    res["detail"]["neg"] = neg

    if n < MIN_SAMPLES:
        warnings.append(f"not enough samples for cold start: n={n} < {MIN_SAMPLES}")
        res["detail"]["reason"] = "min_samples"
        return res
    if pos < MIN_POS:
        warnings.append(f"not enough positive labels for cold start: pos={pos} < {MIN_POS}")
        res["detail"]["reason"] = "min_pos"
        return res
    if pos == 0 or neg == 0:
        warnings.append("only one class present (all 0 or all 1): training skipped.")
        res["detail"]["reason"] = "single_class"
        return res

    models_dir = _get_models_dir(s)
    res["models_dir"] = str(models_dir)

    lr_model = _train_lr(train_df, warnings)
    lr_path = models_dir / LR_MODEL_PATH.name
    lr_saved = _save_joblib_model(lr_model, lr_path, warnings)
    res["lr_saved"] = bool(lr_saved)

    lgbm_model = _train_lgbm(train_df, warnings)
    lgbm_path = models_dir / LGBM_MODEL_PATH.name
    lgbm_saved = _save_joblib_model(lgbm_model, lgbm_path, warnings)
    res["lgbm_saved"] = bool(lgbm_saved)

    res["trained"] = bool(lr_saved or lgbm_saved)
    res["detail"]["lr_path"] = str(lr_path) if lr_saved else ""
    res["detail"]["lgbm_path"] = str(lgbm_path) if lgbm_saved else ""
    res["detail"]["reason"] = "ok" if res["trained"] else "save_failed"
    return res


# ============================================================
# Markdown helpers
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
    topn = report.get("topn", 10)
    lookback_days = report.get("lookback_days", 150)
    today = report.get("today_yyyymmdd", "")
    latest_snapshot = report.get("latest_snapshot_yyyymmdd", "")
    upper_bound = report.get("label_upper_bound_yyyymmdd", "")
    latest_hit = report.get("latest_hit")
    train_meta = report.get("train_meta", {}) or {}
    train_result = report.get("train_result", {}) or {}
    warnings = report.get("warnings", []) or []
    hit_rows_done_last10 = report.get("hit_rows_done_last10", []) or []

    md_lines: List[str] = []
    md_lines.append("# Step7 自学习报告（latest）")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{report.get('ts','')}")
    md_lines.append(f"- Today：{today}")
    md_lines.append(f"- LatestSnapshot：{latest_snapshot or 'N/A'}")
    md_lines.append(f"- LabelUpperBound：{upper_bound}")
    md_lines.append(f"- TopN：{topn}")
    md_lines.append(f"- Lookback：{lookback_days} 天")
    md_lines.append("")

    md_lines.append("## 1) 最新命中")
    if latest_hit:
        md_lines.append("")
        md_lines.append(f"- trade_date：{latest_hit.get('trade_date','')}")
        md_lines.append(f"- expected_next_trade_date：{latest_hit.get('expected_next_trade_date','')}")
        md_lines.append(f"- actual_next_trade_date：{latest_hit.get('actual_next_trade_date','')}")
        md_lines.append(f"- hit/topn：{latest_hit.get('hit','')}/{latest_hit.get('topn','')}")
        md_lines.append(f"- hit_rate：{latest_hit.get('hit_rate','')}")
        if latest_hit.get("note"):
            md_lines.append(f"- note：{latest_hit.get('note','')}")
    else:
        md_lines.append("")
        md_lines.append("- 暂无可验证命中（对照日快照尚未产生，或尚未形成有效对照）")

    md_lines.append("")
    md_lines.append("## 1.1) 近10日 Top10 命中率（done-only）")
    md_lines.append("")
    if hit_rows_done_last10:
        md_lines.append(_md_table(
            hit_rows_done_last10,
            cols=["trade_date", "actual_next_trade_date", "topn", "hit", "hit_rate"]
        ))
    else:
        md_lines.append("- 暂无近10日可统计数据（可能全部为 pending，或尚未形成有效对照）")

    md_lines.append("")
    md_lines.append("## 2) 训练数据概况")
    md_lines.append("")
    md_lines.append(f"- 特征历史文件：{train_meta.get('feature_history_file','') or '未找到'}")
    md_lines.append(f"- 原始行数：{train_meta.get('rows_raw',0)}")
    md_lines.append(f"- 过滤后行数：{train_meta.get('rows_used',0)}")
    md_lines.append(f"- 丢弃全零特征行：{train_meta.get('rows_dropped_allzero',0)}")
    md_lines.append(f"- 日期总数：{train_meta.get('dates_total',0)}")
    md_lines.append(f"- 使用日期：{train_meta.get('dates_used',0)}")

    md_lines.append("")
    md_lines.append("## 3) 训练执行结果")
    md_lines.append("")
    md_lines.append(f"- trained：{train_result.get('trained')}")
    md_lines.append(f"- lr_saved：{train_result.get('lr_saved')}")
    md_lines.append(f"- lgbm_saved：{train_result.get('lgbm_saved')}")
    md_lines.append(f"- models_dir：{train_result.get('models_dir','')}")
    if isinstance(train_result.get("detail"), dict):
        d = train_result["detail"]
        md_lines.append(f"- train_rows：{d.get('train_rows','')}")
        md_lines.append(f"- pos/neg：{d.get('pos','')}/{d.get('neg','')}")
        if d.get("reason"):
            md_lines.append(f"- reason：{d.get('reason')}")
        if d.get("lr_path"):
            md_lines.append(f"- lr_path：{d.get('lr_path')}")
        if d.get("lgbm_path"):
            md_lines.append(f"- lgbm_path：{d.get('lgbm_path')}")

    if warnings:
        md_lines.append("")
        md_lines.append("## 4) Warnings")
        md_lines.append("")
        for w in warnings[:80]:
            md_lines.append(f"- {w}")
        if len(warnings) > 80:
            md_lines.append(f"- ...（共 {len(warnings)} 条，仅展示前 80 条）")

    return "\n".join(md_lines) + "\n"


# ============================================================
# Train pipeline
# ============================================================
def train_models_pipeline(
    s: Settings,
    outputs_dir: Path,
    lookback_days: int,
    sampling_state: Dict[str, Any],
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    train_df, train_meta = _build_train_set_from_feature_history(
        outputs_dir=outputs_dir,
        lookback_days=lookback_days,
        warnings=warnings,
    )

    if bool(sampling_state.get("quality_gate_pass")):
        train_result = _train_and_save_models(s=s, train_df=train_df, warnings=warnings)
    else:
        if train_df is None or train_df.empty:
            tr = {"train_rows": 0, "pos": 0, "neg": 0}
        else:
            pos = int(train_df["label"].astype(int).sum())
            neg = int(len(train_df) - pos)
            tr = {"train_rows": int(len(train_df)), "pos": pos, "neg": neg}

        train_result = {
            "trained": False,
            "lr_saved": False,
            "lgbm_saved": False,
            "models_dir": "",
            "detail": {
                **tr,
                "reason": "skip_train: quality_gate_fail",
                "quality_gate_pass": bool(sampling_state.get("quality_gate_pass")),
                "days_covered": int(sampling_state.get("days_covered", 0)),
            },
        }
        warnings.append("skip_train: quality_gate_fail")

    return train_df, train_meta, train_result


# ============================================================
# Main Step7
# ============================================================
def run_step7(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []

    outputs_dir = _get_outputs_dir(s)
    learning_dir = outputs_dir / "learning"
    _ensure_dir(outputs_dir)
    _ensure_dir(learning_dir)

    today = _today_yyyymmdd()
    latest_snapshot = _latest_snapshot_yyyymmdd()
    upper_bound = _upper_bound_yyyymmdd()

    try:
        topn = int(getattr(s, "topn", 10) or 10)
    except Exception:
        topn = 10

    try:
        lookback_days = int(getattr(s, "step7_lookback_days", getattr(s, "lookback_days", 150)) or 150)
    except Exception:
        lookback_days = 150

    # ---------------------------
    # 0) Auto-Sampling + Quality Gate
    # ---------------------------
    fh_raw, fh_path = _read_feature_history(outputs_dir)
    fh_df = _normalize_feature_history_v2(fh_raw) if not fh_raw.empty else pd.DataFrame()
    rows_ser = _rows_per_day_series(fh_df)
    days_covered = int(len(rows_ser)) if len(rows_ser) else 0
    rows_last = [int(v) for v in rows_ser.tail(120).tolist()] if len(rows_ser) else []
    quality_pass, q_details = _quality_gate_v2(fh_df)

    prev_state: Dict[str, Any] = {}
    prev_p = learning_dir / SAMPLING_STATE_FILE
    if prev_p.exists():
        try:
            prev_state = json.loads(prev_p.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}

    prev_stage = str(prev_state.get("sampling_stage", "S1_MVP"))
    stage, target_rows, stage_debug = _decide_sampling_stage(
        prev_stage=prev_stage,
        days_covered=days_covered,
        rows_last=rows_last,
        quality_pass=bool(quality_pass),
    )

    sampling_state = {
        "sampling_stage": stage,
        "target_rows_per_day": int(target_rows),
        "days_covered": int(days_covered),
        "rows_per_day_last_N": rows_last[-120:],
        "quality_gate_pass": bool(quality_pass),
        "quality_gate_details": q_details,
        "stage_debug": stage_debug,
        "pseudo_ratio": float(q_details.get("pseudo_ratio", 1.0)),
        "feature_history_file": fh_path,
        "updated_at_utc": _utc_now_iso(),
        "model_version": str(os.getenv("GITHUB_SHA") or os.getenv("GITHUB_RUN_ID") or ""),
        "today_yyyymmdd": today,
        "latest_snapshot_yyyymmdd": latest_snapshot,
        "label_upper_bound_yyyymmdd": upper_bound,
    }
    sampling_state_path = _write_sampling_state(outputs_dir, sampling_state)

    # ---------------------------
    # 1) 命中率统计（V2 only）
    # ---------------------------
    hit_df, hit_csv, latest_hit = build_hit_history(
        outputs_dir=outputs_dir,
        learning_dir=learning_dir,
        topn=topn,
        upper_bound=upper_bound,
        warnings=warnings,
    )

    hit_rows_all: List[Dict[str, Any]] = []
    hit_rows_done_last10: List[Dict[str, Any]] = []
    if hit_df is not None and not hit_df.empty:
        hit_rows_all = hit_df.to_dict(orient="records")
        d = hit_df.copy()
        d["actual_next_trade_date"] = d.get("actual_next_trade_date", "").astype(str).str.strip()
        d["hit_rate"] = d.get("hit_rate", "").astype(str).str.strip()
        d = d[(d["actual_next_trade_date"] != "") & (d["hit_rate"] != "") & (d["hit_rate"].str.lower() != "nan")]
        if not d.empty:
            d = d.sort_values("trade_date").tail(10)
            hit_rows_done_last10 = d[["trade_date", "actual_next_trade_date", "topn", "hit", "hit_rate"]].to_dict(orient="records")

    # ---------------------------
    # 2) 训练流水线（V2 only）
    # ---------------------------
    train_df, train_meta, train_result = train_models_pipeline(
        s=s,
        outputs_dir=outputs_dir,
        lookback_days=lookback_days,
        sampling_state=sampling_state,
        warnings=warnings,
    )

    # ---------------------------
    # 3) 报告
    # ---------------------------
    report = {
        "ts": _now_str(),
        "topn": topn,
        "lookback_days": lookback_days,
        "today_yyyymmdd": today,
        "latest_snapshot_yyyymmdd": latest_snapshot,
        "label_upper_bound_yyyymmdd": upper_bound,
        "latest_hit": latest_hit,
        "hit_rows_all": hit_rows_all,
        "hit_rows_done_last10": hit_rows_done_last10,
        "train_meta": train_meta,
        "sampling_state": sampling_state,
        "sampling_state_path": sampling_state_path,
        "train_result": train_result,
        "warnings": warnings,
    }

    report_json = learning_dir / "step7_report_latest.json"
    _safe_write_json(report_json, report)

    report_md = learning_dir / "step7_report_latest.md"
    _safe_write_text(report_md, render_report_md(report))

    return {
        "step7_learning": {
            "hit_history_csv": str(hit_csv),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "models_dir": str(train_result.get("models_dir", "")),
            "trained": bool(train_result.get("trained")),
            "train_rows": int(train_result.get("detail", {}).get("train_rows", 0)) if isinstance(train_result.get("detail"), dict) else 0,
            "warnings": warnings,
        }
    }


if __name__ == "__main__":
    s = Settings()
    out = run_step7(s, ctx={})
    print(json.dumps(out, ensure_ascii=False, indent=2))
