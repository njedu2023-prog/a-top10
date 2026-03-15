from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


CODE_COL_CANDIDATES = ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]
NAME_COL_CANDIDATES = ["name", "stock_name", "名称", "股票", "证券名称", "股票简称"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"]
CLOSE_COL_CANDIDATES = ["close", "收盘价", "最新价", "last_close"]
TURNOVER_COL_CANDIDATES = ["turnover_rate", "换手率"]
SEAL_AMOUNT_COL_CANDIDATES = ["seal_amount", "封单额", "封单金额"]
OPEN_TIMES_COL_CANDIDATES = ["open_times", "炸板次数", "开板次数"]


# =========================
# Contracts
# =========================
V3_PRED_BASE_COLS = [
    "rank",
    "ts_code",
    "name",
    "Probability",
    "_prob_src",
    "StrengthScore",
    "ThemeBoost",
    "prob_lr",
    "prob_lgbm",
    "prob_rule",
    "seal_amount",
    "open_times",
    "turnover_rate",
    "close",
    "board",
]
V3_PRED_META_COLS = ["trade_date", "verify_date", "run_id", "run_attempt", "commit_sha", "generated_at_utc"]
V3_PRED_COLS = V3_PRED_META_COLS[:2] + V3_PRED_BASE_COLS + V3_PRED_META_COLS[2:]

FEATURE_HISTORY_COLS = [
    "run_time_utc",
    "trade_date",
    "ts_code",
    "name",
    "Probability",
    "_prob_src",
    "StrengthScore",
    "ThemeBoost",
    "seal_amount",
    "open_times",
    "turnover_rate",
    "prob_lr",
    "prob_lgbm",
    "prob_rule",
    "is_sample_mature",
    "mature_reason",
    "label_delay_flag",
    "y_limit_hit",
    "y_next_ret",
    "learnable_flag",
    "reject_reason",
    "sample_quality_grade",
    "batch_quality_score",
    "gate_version",
    "label_version",
    "verify_date",
    "close",
]

FEATURE_HISTORY_REQUIRED_KEYS = ["trade_date", "ts_code"]
FEATURE_HISTORY_DROP_COLS = [
    "prob_ml",
    "prob_final",
    "prob_src",
    "prob_ml_available",
    "prob_fusion_mode",
]

FEATURE_HISTORY_PRESERVE_IF_NEW_EMPTY = {
    "is_sample_mature",
    "mature_reason",
    "label_delay_flag",
    "y_limit_hit",
    "y_next_ret",
    "learnable_flag",
    "reject_reason",
    "sample_quality_grade",
    "batch_quality_score",
    "gate_version",
    "label_version",
    "verify_date",
    "close",
}


# =========================
# basic utils
# =========================
def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_run_meta() -> Dict[str, str]:
    run_id = (os.getenv("RUN_ID") or os.getenv("GITHUB_RUN_ID") or "").strip()
    run_attempt = (os.getenv("RUN_ATTEMPT") or os.getenv("GITHUB_RUN_ATTEMPT") or "").strip()
    sha = (os.getenv("COMMIT_SHA") or os.getenv("GITHUB_SHA") or "").strip()

    if not run_id:
        run_id = datetime.utcnow().strftime("ts%Y%m%d%H%M%S")
    if not run_attempt:
        run_attempt = "1"

    return {
        "run_id": run_id,
        "run_attempt": run_attempt,
        "commit_sha": sha,
        "generated_at_utc": _utc_now_iso(),
    }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _clean_date_value(x: Any) -> str:
    s = _safe_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d{8}", s):
        return s
    if re.fullmatch(r"\d{8}\.0", s):
        return s.split(".")[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return s


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_df(x: Any) -> Optional[pd.DataFrame]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception:
        return None


def _json_default(o: Any) -> str:
    try:
        return str(o)
    except Exception:
        return repr(o)


def _read_csv_guess(p: Path) -> pd.DataFrame:
    if p is None or not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _write_text_overwrite(path: Path, text: str, *, encoding: str = "utf-8") -> bool:
    _ensure_dir(path.parent)
    path.write_text(text, encoding=encoding)
    print(f"[WRITE] {path} (overwrite)")
    return True


def _write_csv_once(df: pd.DataFrame, path: Path) -> bool:
    _ensure_dir(path.parent)
    if path.exists():
        print(f"[SKIP] exists (write-once): {path}")
        return False
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path} rows={len(df)}")
    return True


def _write_csv_overwrite(df: pd.DataFrame, path: Path) -> bool:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path} rows={len(df)} (overwrite)")
    return True


# =========================
# market / calendar helpers
# =========================
def _ctx_df(ctx: Any, keys: Sequence[str]) -> Optional[pd.DataFrame]:
    if not isinstance(ctx, dict):
        return None
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


def _ctx_path(ctx: Any, keys: Sequence[str]) -> Optional[Path]:
    if not isinstance(ctx, dict):
        return None
    for k in keys:
        v = ctx.get(k)
        if v is None:
            continue
        try:
            return Path(v)
        except Exception:
            continue
    return None


def _ctx_trade_date(ctx: Any) -> str:
    if not isinstance(ctx, dict):
        return ""
    for k in ("trade_date", "TRADE_DATE", "asof", "date", "snapshot_date"):
        s = _safe_str(ctx.get(k))
        if len(s) == 8 and s.isdigit():
            return s
    return ""


def _norm_code(code: Any) -> Tuple[str, str]:
    s = _safe_str(code).upper()
    if not s:
        return "", ""
    code6 = s.split(".")[0]
    if len(code6) > 6 and code6.isdigit():
        code6 = code6[-6:]
    return s, code6


def _build_code_sets(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[str, set, set]:
    ts_set: set = set()
    c6_set: set = set()
    code_col = _first_existing_col(df, candidates)
    if code_col:
        for v in df[code_col].tolist():
            ts, c6 = _norm_code(v)
            if ts:
                ts_set.add(ts)
            if c6:
                c6_set.add(c6)
    return code_col or "", ts_set, c6_set


def _count_limitups(limit_df: Optional[pd.DataFrame]) -> int:
    if limit_df is None or limit_df.empty:
        return 0
    code_col = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not code_col:
        return int(len(limit_df))
    uniq: set = set()
    for v in limit_df[code_col].tolist():
        ts, c6 = _norm_code(v)
        if ts:
            uniq.add(ts)
        elif c6:
            uniq.add(c6)
    return int(len(uniq)) if uniq else int(len(limit_df))


def _resolve_snapshot_dir(settings, ctx, trade_date: str) -> Optional[Path]:
    cands: List[Path] = []

    p = _ctx_path(ctx, ["snapshot_dir", "snap_dir", "snapshot_path"])
    if p:
        cands.append(p)

    sio = getattr(settings, "io", None)
    if sio is not None:
        for attr in ("snapshot_dir", "snapshots_dir", "snapshot_root", "snapshot_base", "snapshots_root"):
            v = getattr(sio, attr, None)
            if v:
                try:
                    cands.append(Path(v))
                except Exception:
                    pass

    dr = getattr(settings, "data_repo", None)
    if dr is not None:
        for attr in ("snapshot_dir", "snapshot_path", "get_snapshot_dir"):
            fn = getattr(dr, attr, None)
            if callable(fn):
                try:
                    cands.append(Path(fn(trade_date)))
                except Exception:
                    pass
        for attr in ("root", "warehouse_root"):
            v = getattr(dr, attr, None)
            if v:
                try:
                    cands.append(Path(v))
                except Exception:
                    pass

    for base in cands:
        try:
            b = Path(base)
        except Exception:
            continue
        if not b.exists():
            continue
        if trade_date in str(b):
            return b
        if (b / trade_date).exists():
            return b / trade_date
        y = trade_date[:4]
        for cand in (b / y / trade_date, b / "snapshots" / trade_date, b / "data" / "raw" / y / trade_date, b / "raw" / y / trade_date):
            if cand.exists():
                return cand

    for guess in (Path("data_repo/snapshots"), Path("data_repo"), Path("snapshots"), Path("_warehouse")):
        if (guess / trade_date).exists():
            return guess / trade_date
    return None


def _load_limit_df(settings, ctx, trade_date: str) -> pd.DataFrame:
    ctx_td = _ctx_trade_date(ctx)
    if ctx_td and ctx_td == trade_date:
        df = _ctx_df(ctx, ["limit_df", "limit_list", "limit", "limit_list_d", "limit_up", "limitup"])
        if df is not None and not df.empty:
            return df.copy()

    snap = _resolve_snapshot_dir(settings, ctx, trade_date)
    if snap is None:
        return pd.DataFrame()

    for fn in ("limit_list_d.csv", "limit_list.csv", "limit_listd.csv", "limit_list_d"):
        p = snap / fn
        if p.exists():
            return _read_csv_guess(p)
    return pd.DataFrame()


def _list_trade_dates(settings) -> List[str]:
    dates: List[str] = []
    dr = getattr(settings, "data_repo", None)
    if dr is not None:
        fn = getattr(dr, "list_snapshot_dates", None)
        if callable(fn):
            try:
                dates = list(fn()) or []
            except Exception:
                pass
    return sorted([d for d in dates if isinstance(d, str) and len(d) == 8 and d.isdigit()])


def _prev_next_trade_date(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    if calendar and trade_date in calendar:
        i = calendar.index(trade_date)
        prev_td = calendar[i - 1] if i - 1 >= 0 else ""
        next_td = calendar[i + 1] if i + 1 < len(calendar) else ""
        return prev_td, next_td
    return "", ""


def _prev_next_trade_date_with_fallback(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    prev_td, next_td = _prev_next_trade_date(calendar, trade_date)
    if prev_td and next_td:
        return prev_td, next_td

    try:
        d = datetime.strptime(trade_date, "%Y%m%d")
    except Exception:
        return prev_td or "", next_td or ""

    prev_d = d - timedelta(days=1)
    while prev_d.weekday() >= 5:
        prev_d -= timedelta(days=1)

    next_d = d + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)

    if not prev_td:
        prev_td = prev_d.strftime("%Y%m%d")
    if not next_td:
        next_td = next_d.strftime("%Y%m%d")
    return prev_td, next_td


# =========================
# frame normalization
# =========================
def _normalize_rank(df: pd.DataFrame) -> pd.Series:
    if "rank" in df.columns:
        s = pd.to_numeric(df["rank"], errors="coerce")
        if s.notna().any():
            return s
    if "排名" in df.columns:
        s = pd.to_numeric(df["排名"], errors="coerce")
        if s.notna().any():
            return s
    return pd.Series(range(1, len(df) + 1), index=df.index, dtype="int64")


def _extract_numeric(df: pd.DataFrame, candidates: Sequence[str], default: Any = "") -> pd.Series:
    col = _first_existing_col(df, candidates)
    if not col:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _extract_text(df: pd.DataFrame, candidates: Sequence[str], default: str = "") -> pd.Series:
    col = _first_existing_col(df, candidates)
    if not col:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def _canonicalize_prediction_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    V3 预测产物标准化：主概率轴只认 Probability，来源只认 _prob_src。
    不再让 prob_ml / prob_final / prob_src 进入正式产物契约。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=V3_PRED_BASE_COLS)

    src = df.copy()
    out = pd.DataFrame(index=src.index)

    code_col = _first_existing_col(src, CODE_COL_CANDIDATES)
    name_col = _first_existing_col(src, NAME_COL_CANDIDATES)
    board_col = _first_existing_col(src, BOARD_COL_CANDIDATES)

    out["rank"] = _normalize_rank(src)
    out["ts_code"] = src[code_col] if code_col else ""
    out["name"] = src[name_col] if name_col else ""
    out["board"] = src[board_col] if board_col else ""

    if "Probability" in src.columns:
        out["Probability"] = src["Probability"]
    elif "prob_final" in src.columns:
        out["Probability"] = src["prob_final"]
    elif "prob_ml" in src.columns:
        out["Probability"] = src["prob_ml"]
    else:
        out["Probability"] = ""

    if "_prob_src" in src.columns:
        out["_prob_src"] = src["_prob_src"]
    elif "prob_src" in src.columns:
        out["_prob_src"] = src["prob_src"]
    else:
        out["_prob_src"] = ""

    out["StrengthScore"] = src["StrengthScore"] if "StrengthScore" in src.columns else (src["强度得分"] if "强度得分" in src.columns else "")
    out["ThemeBoost"] = src["ThemeBoost"] if "ThemeBoost" in src.columns else (src["题材加成"] if "题材加成" in src.columns else "")
    out["prob_lr"] = src["prob_lr"] if "prob_lr" in src.columns else ""
    out["prob_lgbm"] = src["prob_lgbm"] if "prob_lgbm" in src.columns else ""
    out["prob_rule"] = src["prob_rule"] if "prob_rule" in src.columns else ""
    out["seal_amount"] = _extract_numeric(src, SEAL_AMOUNT_COL_CANDIDATES)
    out["open_times"] = _extract_numeric(src, OPEN_TIMES_COL_CANDIDATES)
    out["turnover_rate"] = _extract_numeric(src, TURNOVER_COL_CANDIDATES)
    out["close"] = _extract_numeric(src, CLOSE_COL_CANDIDATES)

    for col in ["Probability", "StrengthScore", "ThemeBoost", "prob_lr", "prob_lgbm", "prob_rule"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out = out.reindex(columns=V3_PRED_BASE_COLS, fill_value="")
    return out


def _enrich_prediction_with_meta(df: Optional[pd.DataFrame], trade_date: str, verify_date: str, run_meta: Dict[str, str]) -> pd.DataFrame:
    base = _canonicalize_prediction_frame(df)
    out = base.copy()
    out.insert(0, "trade_date", trade_date)
    out.insert(1, "verify_date", verify_date)
    out["run_id"] = run_meta["run_id"]
    out["run_attempt"] = run_meta["run_attempt"]
    out["commit_sha"] = run_meta["commit_sha"]
    out["generated_at_utc"] = run_meta["generated_at_utc"]
    out = out.reindex(columns=V3_PRED_COLS, fill_value="")
    return out


def _canonicalize_feature_history_batch(
    df: Optional[pd.DataFrame],
    *,
    trade_date: str,
    verify_date: str,
    run_time_utc: str,
    ctx: Any,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_HISTORY_COLS)

    src = df.copy()
    out = pd.DataFrame(index=src.index)

    code_col = _first_existing_col(src, CODE_COL_CANDIDATES)
    name_col = _first_existing_col(src, NAME_COL_CANDIDATES)

    out["run_time_utc"] = run_time_utc
    out["trade_date"] = trade_date
    out["ts_code"] = src[code_col] if code_col else ""
    out["name"] = src[name_col] if name_col else ""

    if "Probability" in src.columns:
        out["Probability"] = src["Probability"]
    elif "prob_final" in src.columns:
        out["Probability"] = src["prob_final"]
    elif "prob_ml" in src.columns:
        out["Probability"] = src["prob_ml"]
    else:
        out["Probability"] = ""

    if "_prob_src" in src.columns:
        out["_prob_src"] = src["_prob_src"]
    elif "prob_src" in src.columns:
        out["_prob_src"] = src["prob_src"]
    else:
        out["_prob_src"] = ""

    out["StrengthScore"] = src["StrengthScore"] if "StrengthScore" in src.columns else ""
    out["ThemeBoost"] = src["ThemeBoost"] if "ThemeBoost" in src.columns else ""
    out["seal_amount"] = _extract_numeric(src, SEAL_AMOUNT_COL_CANDIDATES)
    out["open_times"] = _extract_numeric(src, OPEN_TIMES_COL_CANDIDATES)
    out["turnover_rate"] = _extract_numeric(src, TURNOVER_COL_CANDIDATES)
    out["prob_lr"] = src["prob_lr"] if "prob_lr" in src.columns else ""
    out["prob_lgbm"] = src["prob_lgbm"] if "prob_lgbm" in src.columns else ""
    out["prob_rule"] = src["prob_rule"] if "prob_rule" in src.columns else ""

    # step7 labels / quality fields: writers 不制造假值，只做稳定承接
    for col in [
        "is_sample_mature",
        "mature_reason",
        "label_delay_flag",
        "y_limit_hit",
        "y_next_ret",
        "learnable_flag",
        "reject_reason",
        "sample_quality_grade",
        "batch_quality_score",
        "gate_version",
        "label_version",
    ]:
        out[col] = src[col] if col in src.columns else ""

    out["verify_date"] = src["verify_date"] if "verify_date" in src.columns else verify_date
    out["close"] = _extract_numeric(src, CLOSE_COL_CANDIDATES)

    # 若 full_df 无 close，尝试从上下文行情表回填
    if out["close"].isna().all():
        close_map = _build_close_map_from_ctx(ctx)
        if close_map:
            out["close"] = [close_map.get(_safe_str(v), pd.NA) for v in out["ts_code"]]

    for col in [
        "Probability",
        "StrengthScore",
        "ThemeBoost",
        "seal_amount",
        "open_times",
        "turnover_rate",
        "prob_lr",
        "prob_lgbm",
        "prob_rule",
        "is_sample_mature",
        "label_delay_flag",
        "y_limit_hit",
        "y_next_ret",
        "learnable_flag",
        "batch_quality_score",
        "close",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop(columns=[c for c in FEATURE_HISTORY_DROP_COLS if c in out.columns], errors="ignore")
    out = out.reindex(columns=FEATURE_HISTORY_COLS, fill_value="")
    out["trade_date"] = out["trade_date"].map(_clean_date_value)
    out["verify_date"] = out["verify_date"].map(_clean_date_value)
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out = out[out["ts_code"] != ""].copy()
    out = out.drop_duplicates(subset=FEATURE_HISTORY_REQUIRED_KEYS, keep="last").reset_index(drop=True)
    return out


# =========================
# feature_history merge logic
# =========================
def _build_close_map_from_ctx(ctx: Any) -> Dict[str, Any]:
    cands = [
        _ctx_df(ctx, ["daily_df", "daily", "daily_basic", "quote_df", "quotes"]),
        _ctx_df(ctx, ["limit_df", "limit_list", "limit_list_d"]),
    ]
    close_map: Dict[str, Any] = {}
    for df in cands:
        if df is None or df.empty:
            continue
        code_col = _first_existing_col(df, CODE_COL_CANDIDATES)
        close_col = _first_existing_col(df, CLOSE_COL_CANDIDATES)
        if not code_col or not close_col:
            continue
        for _, row in df[[code_col, close_col]].dropna(subset=[code_col]).iterrows():
            close_map[_safe_str(row[code_col])] = row[close_col]
    return close_map


def _normalize_existing_feature_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_HISTORY_COLS)

    d = df.copy()

    if "Probability" not in d.columns:
        if "prob_final" in d.columns:
            d["Probability"] = d["prob_final"]
        elif "prob_ml" in d.columns:
            d["Probability"] = d["prob_ml"]
        else:
            d["Probability"] = ""

    if "_prob_src" not in d.columns:
        if "prob_src" in d.columns:
            d["_prob_src"] = d["prob_src"]
        else:
            d["_prob_src"] = ""

    for col in FEATURE_HISTORY_COLS:
        if col not in d.columns:
            d[col] = ""

    d = d.drop(columns=[c for c in FEATURE_HISTORY_DROP_COLS if c in d.columns], errors="ignore")
    d = d.reindex(columns=FEATURE_HISTORY_COLS, fill_value="")
    d["trade_date"] = d["trade_date"].map(_clean_date_value)
    d["verify_date"] = d["verify_date"].map(_clean_date_value)
    d["ts_code"] = d["ts_code"].astype(str).str.strip()
    d = d[(d["trade_date"] != "") & (d["ts_code"] != "")].copy()
    d = d.drop_duplicates(subset=FEATURE_HISTORY_REQUIRED_KEYS, keep="last").reset_index(drop=True)
    return d


def _merge_feature_history(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    existing_df = _normalize_existing_feature_history(existing_df)
    new_df = _normalize_existing_feature_history(new_df)

    if existing_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = existing_df.copy()
    else:
        existing_map = {
            (str(r["trade_date"]), str(r["ts_code"])): r.to_dict()
            for _, r in existing_df.iterrows()
        }

        for _, row in new_df.iterrows():
            key = (str(row["trade_date"]), str(row["ts_code"]))
            new_row = row.to_dict()
            old_row = existing_map.get(key)
            if old_row is None:
                existing_map[key] = new_row
                continue

            merged_row = old_row.copy()
            for col in FEATURE_HISTORY_COLS:
                new_val = new_row.get(col)
                old_val = old_row.get(col)
                if col in FEATURE_HISTORY_PRESERVE_IF_NEW_EMPTY and _is_missing(new_val) and not _is_missing(old_val):
                    merged_row[col] = old_val
                elif not _is_missing(new_val):
                    merged_row[col] = new_val
                else:
                    merged_row[col] = old_val
            existing_map[key] = merged_row

        merged = pd.DataFrame(existing_map.values())

    for col in FEATURE_HISTORY_COLS:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged.reindex(columns=FEATURE_HISTORY_COLS, fill_value="")

    merged["trade_date"] = merged["trade_date"].map(_clean_date_value)
    merged["verify_date"] = merged["verify_date"].map(_clean_date_value)
    merged["ts_code"] = merged["ts_code"].astype(str).str.strip()
    merged["run_time_utc"] = merged["run_time_utc"].astype(str)

    merged = merged.sort_values(by=["trade_date", "ts_code", "run_time_utc"], ascending=[True, True, True], na_position="last")
    merged = merged.drop_duplicates(subset=FEATURE_HISTORY_REQUIRED_KEYS, keep="last").reset_index(drop=True)
    return merged


# =========================
# markdown helpers
# =========================
def _df_to_md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    if df is None or df.empty:
        return ""

    d = df.copy()
    if cols is not None:
        use_cols = [c for c in cols if c in d.columns]
        if use_cols:
            d = d[use_cols].copy()

    col_map = {
        "rank": "排名",
        "rank_limit": "排名",
        "ts_code": "代码",
        "name": "股票",
        "Probability": "Probability",
        "_prob_src": "_prob_src",
        "StrengthScore": "强度得分",
        "ThemeBoost": "题材加成",
        "board": "板块",
        "命中": "命中",
        "日期": "日期",
        "命中数": "命中数",
        "命中率": "命中率",
        "当日涨停家数": "当日涨停家数",
    }
    d = d.rename(columns=col_map)

    try:
        return d.to_markdown(index=False)
    except Exception:
        x = d.copy().fillna("")
        headers = list(x.columns)

        def esc(v: Any) -> str:
            s = str(v)
            s = s.replace("\n", " ").replace("\r", " ").replace("|", "\\|")
            return s

        lines = []
        lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for _, row in x.iterrows():
            lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
        return "\n".join(lines)


# =========================
# report helpers
# =========================
def _load_json_topn(outdir: Path, td: str) -> Optional[pd.DataFrame]:
    if not td:
        return None
    p = outdir / f"predict_top10_{td}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw_topn = payload.get("topN") or []
    return _to_df(raw_topn)


def _topn_to_hit_df(topn_df: Optional[pd.DataFrame], limit_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    pred = _canonicalize_prediction_frame(topn_df)

    if pred.empty:
        metrics.update({"hit_count": 0, "top_count": 0, "limit_count": _count_limitups(limit_df), "hit_rate": ""})
        return pd.DataFrame(), metrics

    _, limit_ts, limit_c6 = _build_code_sets(limit_df, CODE_COL_CANDIDATES)
    hits: List[str] = []
    hit_count = 0
    for v in pred["ts_code"].tolist():
        ts, c6 = _norm_code(v)
        ok = (ts in limit_ts) or (c6 in limit_c6)
        hits.append("是" if ok else "否")
        if ok:
            hit_count += 1

    out = pred.copy()
    out["命中"] = hits
    top_count = int(len(out))
    metrics.update({
        "hit_count": int(hit_count),
        "top_count": top_count,
        "limit_count": int(_count_limitups(limit_df)),
        "hit_rate": f"{round(hit_count * 100.0 / float(top_count))}%" if top_count > 0 else "",
    })
    return out, metrics


def _standardize_strength_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    pred = _canonicalize_prediction_frame(df)
    if pred.empty:
        return pd.DataFrame()

    out = pred[["rank", "ts_code", "name", "Probability", "StrengthScore", "ThemeBoost", "board"]].copy()
    out = out.sort_values(by=["StrengthScore", "Probability"], ascending=False, na_position="last").reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def _join_limit_strength(limit_df: pd.DataFrame, full_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if limit_df is None or limit_df.empty:
        return None
    lcode = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not lcode:
        return None

    l = limit_df[[lcode]].copy()
    if lcode != "ts_code":
        l["ts_code"] = l[lcode]

    pred = _canonicalize_prediction_frame(full_df)
    if pred.empty:
        out = l[["ts_code"]].copy()
        out["name"] = ""
        out["Probability"] = ""
        out["_prob_src"] = ""
        out["StrengthScore"] = ""
        out["ThemeBoost"] = ""
        out["board"] = ""
        out.insert(0, "rank", range(1, len(out) + 1))
        return out

    merged = pd.merge(l[["ts_code"]], pred, on="ts_code", how="left")
    merged = merged.sort_values(by=["StrengthScore", "Probability"], ascending=False, na_position="last")
    merged = merged.reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    return merged


def _recent_hit_history(outdir: Path, settings, ctx, max_days: int = 10) -> pd.DataFrame:
    calendar = _list_trade_dates(settings)
    files = sorted(outdir.glob("predict_top10_*.json"), key=lambda p: p.name, reverse=True)
    rows: List[Dict[str, Any]] = []

    for f in files:
        m = re.match(r"predict_top10_(\d{8})\.json", f.name)
        if not m:
            continue
        pred_date = m.group(1)
        _, verify_date = _prev_next_trade_date(calendar, pred_date)
        if not verify_date:
            continue

        topn_df = _load_json_topn(outdir, pred_date)
        limit_df_verify = _load_limit_df(settings, ctx, verify_date)
        _, metrics = _topn_to_hit_df(topn_df, limit_df_verify)
        rows.append({
            "日期": pred_date,
            "命中数": int(metrics.get("hit_count") or 0),
            "命中率": str(metrics.get("hit_rate") or ""),
            "当日涨停家数": _count_limitups(limit_df_verify),
        })
        if len(rows) >= max_days:
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =========================
# V3 history rebuild / write
# =========================
def _write_feature_history(learning_dir: Path, batch_df: pd.DataFrame) -> pd.DataFrame:
    history_path = learning_dir / "feature_history.csv"
    existing_df = _read_csv_guess(history_path)
    merged_df = _merge_feature_history(existing_df, batch_df)
    _write_csv_overwrite(merged_df, history_path)
    return merged_df


# =========================
# main writer
# =========================
def write_outputs(settings, trade_date: str, ctx, gate, topn, learn) -> None:
    outdir = getattr(getattr(settings, "io", None), "outputs_dir", None) or "outputs"
    outdir = Path(outdir)
    _ensure_dir(outdir)

    run_meta = _get_run_meta()

    learning_dir = outdir / "learning"
    warehouse_dir = outdir / "_warehouse" / "pred_top10"
    decisio_dir = outdir / "decisio"

    _ensure_dir(learning_dir)
    _ensure_dir(warehouse_dir)
    _ensure_dir(decisio_dir)

    topn_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None
    limit_up_table_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        raw_topn = None
        for k in ["topN", "topn", "TopN", "top"]:
            if k in topn and topn.get(k) is not None:
                raw_topn = topn.get(k)
                break
        topn_df = _to_df(raw_topn)
        full_df = _to_df(topn.get("full")) if "full" in topn else None
        limit_up_table_df = _to_df(topn.get("limit_up_table")) if "limit_up_table" in topn else None
    else:
        topn_df = _to_df(topn)

    calendar = _list_trade_dates(settings)
    prev_td, next_td = _prev_next_trade_date_with_fallback(calendar, trade_date)

    limit_df_current = _load_limit_df(settings, ctx, trade_date)
    _, metrics_same_day = _topn_to_hit_df(topn_df, limit_df_current)

    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "verify_date": next_td,
        "gate": gate,
        "topN": [] if topn_df is None else topn_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "limit_up_table": [] if limit_up_table_df is None else limit_up_table_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics_same_day,
        "run_meta": run_meta,
    }

    # -----------------
    # JSON / MD
    # -----------------
    json_path = outdir / f"predict_top10_{trade_date}.json"
    _write_text_overwrite(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    md_lines: List[str] = [f"# {trade_date} 预测报告\n"]
    md_lines.append(f"## 《{trade_date} 预测：{next_td} 涨停 TOP 10》\n")

    topn_v3 = _canonicalize_prediction_frame(topn_df)
    if topn_v3.empty:
        reason = ""
        if isinstance(gate, dict):
            msg = _safe_str(gate.get("reason") or gate.get("msg") or "")
            if msg:
                reason = f"（{msg}）"
        md_lines.append(f"⚠️ Gate 未通过，Top10 为空。{reason}\n")
    else:
        md_lines.append(_df_to_md_table(topn_v3, cols=["rank", "ts_code", "name", "Probability", "_prob_src", "StrengthScore", "ThemeBoost", "board"]))
        md_lines.append("")

    md_lines.append(f"## 《{trade_date} 所有涨停股票的强度列表》\n")
    strength_limit_df = None
    if limit_up_table_df is not None and not limit_up_table_df.empty:
        strength_limit_df = _standardize_strength_table(limit_up_table_df)
    else:
        joined = _join_limit_strength(limit_df_current, full_df)
        if joined is not None and not joined.empty:
            strength_limit_df = _standardize_strength_table(joined)

    if strength_limit_df is None or strength_limit_df.empty:
        md_lines.append("(未能生成强度列表：limit_list 或 full 排名为空)\n")
    else:
        md_lines.append(_df_to_md_table(strength_limit_df, cols=["rank", "ts_code", "name", "Probability", "StrengthScore", "ThemeBoost", "board"]))
        md_lines.append("")

    prev_title = prev_td if prev_td else "上一交易日"
    md_lines.append(f"## 《{prev_title} 预测：{trade_date} 命中情况》\n")
    prev_topn_df = _load_json_topn(outdir, prev_td)
    prev_hit_df, _prev_metrics = _topn_to_hit_df(prev_topn_df, limit_df_current)
    if prev_hit_df.empty:
        md_lines.append("（未找到上一交易日预测文件或上一交易日 Top10 为空）\n")
    else:
        md_lines.append(_df_to_md_table(prev_hit_df, cols=["ts_code", "name", "Probability", "命中", "board"]))
        md_lines.append("")

    hist10 = _recent_hit_history(outdir, settings, ctx, max_days=10)
    if not hist10.empty:
        md_lines.append("## 《近10日 Top10 命中率》\n")
        md_lines.append(_df_to_md_table(hist10, cols=["日期", "命中数", "命中率", "当日涨停家数"]))
        md_lines.append("")

    md_text = "\n".join(md_lines)
    md_path = outdir / f"predict_top10_{trade_date}.md"
    _write_text_overwrite(md_path, md_text, encoding="utf-8")
    _write_text_overwrite(outdir / "latest.md", md_text, encoding="utf-8")

    # -----------------
    # prediction csv outputs
    # -----------------
    topn_out = _enrich_prediction_with_meta(topn_df, trade_date, next_td, run_meta)
    full_out = _enrich_prediction_with_meta(full_df, trade_date, next_td, run_meta)

    _write_csv_overwrite(topn_out, learning_dir / f"pred_top10_{trade_date}.csv")
    _write_csv_overwrite(topn_out, learning_dir / "pred_top10_latest.csv")
    _write_csv_once(topn_out, warehouse_dir / f"pred_top10_{trade_date}_{run_meta['run_id']}.csv")

    _write_csv_overwrite(full_out, decisio_dir / f"pred_decisio_{trade_date}.csv")
    _write_csv_overwrite(full_out, decisio_dir / "pred_decisio_latest.csv")

    # -----------------
    # feature_history.csv: V3 正式样本契约
    # -----------------
    feature_source = full_df if full_df is not None and not full_df.empty else topn_df
    feature_batch = _canonicalize_feature_history_batch(
        feature_source,
        trade_date=trade_date,
        verify_date=next_td,
        run_time_utc=run_meta["generated_at_utc"],
        ctx=ctx,
    )
    merged_history = _write_feature_history(learning_dir, feature_batch)

    # -----------------
    # last run marker
    # -----------------
    last_run = learning_dir / "_last_run.txt"
    last_run.write_text(
        (
            f"trade_date={trade_date}\n"
            f"verify_date={next_td}\n"
            f"run_id={run_meta['run_id']}\n"
            f"run_attempt={run_meta['run_attempt']}\n"
            f"commit_sha={run_meta['commit_sha']}\n"
            f"generated_at_utc={run_meta['generated_at_utc']}\n"
            f"feature_history_rows={len(merged_history)}\n"
            f"feature_batch_rows={len(feature_batch)}\n"
        ),
        encoding="utf-8",
    )
    print(f"[WRITE] {last_run} (latest)")

    print(f"✅ V3 outputs written: {md_path}")
