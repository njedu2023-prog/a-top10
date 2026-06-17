from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


CODE_COL_CANDIDATES = ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]
NAME_COL_CANDIDATES = ["name", "stock_name", "名称", "股票", "证券名称", "股票简称"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"]
CLOSE_COL_CANDIDATES = ["close", "收盘价", "最新价", "last_close"]
TURNOVER_COL_CANDIDATES = ["turnover_rate", "换手率"]
SEAL_AMOUNT_COL_CANDIDATES = ["seal_amount", "封单额", "封单金额"]
OPEN_TIMES_COL_CANDIDATES = ["open_times", "炸板次数", "开板次数"]

INTRADAY_REPORT_COLS = [
    "final_score_v2",
    "final_score_base",
    "strength_plus_score",
    "intraday_quality_score",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_bonus",
    "intraday_risk_penalty",
    "intraday_hard_risk_flag",
    "risk_level",
    "risk_tags",
    "intraday_data_status",
    "auction_data_status",
]


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
    "final_score_v2",
    "final_score_base",
    "strength_plus_score",
    "intraday_quality_score",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_bonus",
    "intraday_risk_penalty",
    "intraday_hard_risk_flag",
    "risk_level",
    "risk_tags",
    "intraday_data_status",
    "auction_data_status",
    "seal_amount",
    "open_times",
    "turnover_rate",
    "close",
    "board",
]
V3_PRED_META_COLS = ["trade_date", "verify_date", "run_id", "run_attempt", "commit_sha", "generated_at_utc"]
V3_PRED_COLS = V3_PRED_META_COLS[:2] + V3_PRED_BASE_COLS + V3_PRED_META_COLS[2:]

# decision 专用上游源表契约：只服务 top10-decision/data/pred/pred_source_latest.csv
DECISION_SOURCE_BASE_COLS = [
    "rank",
    "ts_code",
    "name",
    "prob",
    "StrengthScore",
    "ThemeBoost",
    "board",
]
DECISION_SOURCE_META_COLS = ["trade_date", "verify_date", "run_id", "run_attempt", "commit_sha", "generated_at_utc"]
DECISION_SOURCE_COLS = DECISION_SOURCE_META_COLS[:2] + DECISION_SOURCE_BASE_COLS + DECISION_SOURCE_META_COLS[2:]

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
    "final_score_base",
    "final_score_v2",
    "strength_plus_score",
    "intraday_quality_score",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_available",
    "auction_available",
    "risk_level",
    "risk_tags",
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

# A 股非交易日补充表：用于 snapshot calendar 不完整时的硬兜底。
# 重要：这里不是用“自然工作日”猜交易日，而是先排除交易所已公布的休市日。
# 可通过环境变量 A_TOP10_EXTRA_CLOSED_DATES 追加，格式：YYYYMMDD,YYYYMMDD...
A_SHARE_CLOSED_DATES = {
    # 2024
    "20240101",
    "20240209", "20240212", "20240213", "20240214", "20240215", "20240216",
    "20240404", "20240405",
    "20240501", "20240502", "20240503",
    "20240610",
    "20240916", "20240917",
    "20241001", "20241002", "20241003", "20241004", "20241007",
    # 2025
    "20250101",
    "20250128", "20250129", "20250130", "20250131", "20250203", "20250204",
    "20250404",
    "20250501", "20250502", "20250505",
    "20250602",
    "20251001", "20251002", "20251003", "20251006", "20251007", "20251008",
    # 2026（上海证券交易所 2026 年度休市安排）
    "20260101", "20260102",
    "20260216", "20260217", "20260218", "20260219", "20260220", "20260223",
    "20260406",
    "20260501", "20260504", "20260505",
    "20260619",
    "20260925",
    "20261001", "20261002", "20261005", "20261006", "20261007",
}


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
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return isinstance(x, str) and x.strip() == ""


def _clean_date_value(x: Any) -> str:
    s = _safe_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d{8}", s):
        return s
    if re.fullmatch(r"\d{8}\.0", s):
        return s.split(".")[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else s


def _first_existing_col(df: Optional[pd.DataFrame], candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(str(name).lower())
        if hit is not None:
            return hit
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


def _ctx_df(ctx: Any, keys: Sequence[str]) -> Optional[pd.DataFrame]:
    if not isinstance(ctx, dict):
        return None
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v.copy()
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
            pass
    return None


def _ctx_trade_date(ctx: Any) -> str:
    if not isinstance(ctx, dict):
        return ""
    for k in ("trade_date", "TRADE_DATE", "asof", "date", "snapshot_date"):
        s = _clean_date_value(ctx.get(k))
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


def _warehouse_snapshot_dir_candidates(trade_date: str) -> List[Path]:
    y = str(trade_date)[:4]
    return [
        Path("_warehouse/a-share-top3-data/data/raw") / y / str(trade_date),
        Path("outputs/_warehouse/a-share-top3-data/data/raw") / y / str(trade_date),
        Path("data/raw") / y / str(trade_date),
        Path("_warehouse/data/raw") / y / str(trade_date),
    ]


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
        for cand in (
            b / y / trade_date,
            b / "snapshots" / trade_date,
            b / "data" / "raw" / y / trade_date,
            b / "raw" / y / trade_date,
        ):
            if cand.exists():
                return cand

    for guess in (
        Path("data_repo/snapshots"),
        Path("data_repo"),
        Path("snapshots"),
        Path("_warehouse"),
        Path("_warehouse/a-share-top3-data/data/raw") / trade_date[:4] / trade_date,
        Path("outputs/_warehouse/a-share-top3-data/data/raw") / trade_date[:4] / trade_date,
        Path("data/raw") / trade_date[:4] / trade_date,
    ):
        gp = Path(guess)
        if gp.exists() and gp.is_dir():
            if gp.name == trade_date:
                return gp
            if (gp / trade_date).exists():
                return gp / trade_date
    return None


def _load_limit_df(settings, ctx, trade_date: str) -> pd.DataFrame:
    trade_date = _clean_date_value(trade_date)
    if not trade_date:
        return pd.DataFrame()

    ctx_td = _ctx_trade_date(ctx)
    if ctx_td and ctx_td == trade_date:
        df = _ctx_df(ctx, ["limit_df", "limit_list", "limit", "limit_list_d", "limit_up", "limitup"])
        if df is not None and not df.empty:
            return df

    for snap in _warehouse_snapshot_dir_candidates(trade_date):
        if not snap.exists():
            continue
        for fn in ("limit_list_d.csv", "limit_list.csv", "limit_listd.csv", "limit_list_d"):
            p = snap / fn
            if p.exists():
                return _read_csv_guess(p)

    snap = _resolve_snapshot_dir(settings, ctx, trade_date)
    if snap is None:
        return pd.DataFrame()
    for fn in ("limit_list_d.csv", "limit_list.csv", "limit_listd.csv", "limit_list_d"):
        p = snap / fn
        if p.exists():
            return _read_csv_guess(p)
    return pd.DataFrame()


def _load_daily_df(settings, ctx, trade_date: str) -> pd.DataFrame:
    trade_date = _clean_date_value(trade_date)
    if not trade_date:
        return pd.DataFrame()

    ctx_td = _ctx_trade_date(ctx)
    if ctx_td and ctx_td == trade_date:
        df = _ctx_df(ctx, ["daily_df", "daily", "quote_df", "quotes"])
        if df is not None and not df.empty:
            return df

    for snap in _warehouse_snapshot_dir_candidates(trade_date):
        if not snap.exists():
            continue
        p = snap / "daily.csv"
        if p.exists():
            return _read_csv_guess(p)

    snap = _resolve_snapshot_dir(settings, ctx, trade_date)
    if snap is None:
        return pd.DataFrame()
    p = snap / "daily.csv"
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
    clean = sorted({_clean_date_value(d) for d in dates if _clean_date_value(d)})
    return [d for d in clean if len(d) == 8 and d.isdigit()]


def _extra_closed_dates_from_env() -> set:
    raw = os.getenv("A_TOP10_EXTRA_CLOSED_DATES") or os.getenv("TOP10_EXTRA_CLOSED_DATES") or ""
    out = set()
    for item in re.split(r"[,;\s]+", raw.strip()):
        d = _clean_date_value(item)
        if len(d) == 8 and d.isdigit():
            out.add(d)
    return out


def _is_a_share_trade_day(dt: datetime) -> bool:
    d = dt.strftime("%Y%m%d")
    if dt.weekday() >= 5:
        return False
    if d in A_SHARE_CLOSED_DATES or d in _extra_closed_dates_from_env():
        return False
    return True


def _scan_prev_next_a_share_trade_date(trade_date: str) -> Tuple[str, str]:
    try:
        d = datetime.strptime(trade_date, "%Y%m%d")
    except Exception:
        return "", ""

    prev_d = d - timedelta(days=1)
    next_d = d + timedelta(days=1)

    # 上限 45 天，避免异常日期造成死循环。
    for _ in range(45):
        if _is_a_share_trade_day(prev_d):
            break
        prev_d -= timedelta(days=1)
    else:
        prev_d = None

    for _ in range(45):
        if _is_a_share_trade_day(next_d):
            break
        next_d += timedelta(days=1)
    else:
        next_d = None

    return (
        prev_d.strftime("%Y%m%d") if prev_d is not None else "",
        next_d.strftime("%Y%m%d") if next_d is not None else "",
    )


def _prev_next_trade_date(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    trade_date = _clean_date_value(trade_date)
    if calendar and trade_date in calendar:
        i = calendar.index(trade_date)
        prev_td = calendar[i - 1] if i - 1 >= 0 else ""
        next_td = calendar[i + 1] if i + 1 < len(calendar) else ""
        return prev_td, next_td
    return "", ""


def _prev_next_trade_date_with_fallback(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    """
    统一交易日锚点解析。

    优先使用 data_repo 的 snapshot calendar；当历史/未来快照不完整时，
    使用 A 股休市日历兜底，而不是用“自然工作日”硬猜。

    核心修复：20260430 的 next_td 必须为 20260506，而不是 20260501。
    """
    trade_date = _clean_date_value(trade_date)
    prev_td, next_td = _prev_next_trade_date(calendar, trade_date)
    fallback_prev, fallback_next = _scan_prev_next_a_share_trade_date(trade_date)

    # 如果 snapshot calendar 只缺一边，用专业日历补齐；如果两边都有，以 snapshot 为准。
    return prev_td or fallback_prev, next_td or fallback_next


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


def _canonicalize_prediction_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
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

    out["Probability"] = src["Probability"] if "Probability" in src.columns else (
        src["prob_final"] if "prob_final" in src.columns else (src["prob_ml"] if "prob_ml" in src.columns else "")
    )
    out["_prob_src"] = src["_prob_src"] if "_prob_src" in src.columns else (src["prob_src"] if "prob_src" in src.columns else "")
    out["StrengthScore"] = src["StrengthScore"] if "StrengthScore" in src.columns else (
        src["强度得分"] if "强度得分" in src.columns else ""
    )
    out["ThemeBoost"] = src["ThemeBoost"] if "ThemeBoost" in src.columns else (
        src["题材加成"] if "题材加成" in src.columns else ""
    )
    out["prob_lr"] = src["prob_lr"] if "prob_lr" in src.columns else ""
    out["prob_lgbm"] = src["prob_lgbm"] if "prob_lgbm" in src.columns else ""
    out["prob_rule"] = src["prob_rule"] if "prob_rule" in src.columns else ""
    for col in INTRADAY_REPORT_COLS:
        out[col] = src[col] if col in src.columns else ""
    out["seal_amount"] = _extract_numeric(src, SEAL_AMOUNT_COL_CANDIDATES)
    out["open_times"] = _extract_numeric(src, OPEN_TIMES_COL_CANDIDATES)
    out["turnover_rate"] = _extract_numeric(src, TURNOVER_COL_CANDIDATES)
    out["close"] = _extract_numeric(src, CLOSE_COL_CANDIDATES)
    for col in [
        "Probability", "StrengthScore", "ThemeBoost", "prob_lr", "prob_lgbm", "prob_rule",
        "final_score_v2", "final_score_base", "strength_plus_score", "intraday_quality_score",
        "intraday_risk_score", "late_withdraw_score", "reseal_score", "open_board_count",
        "auction_strength_score", "auction_real_volume_score", "seal_stability_score",
        "intraday_bonus", "intraday_risk_penalty", "intraday_hard_risk_flag",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    return out.reindex(columns=V3_PRED_BASE_COLS, fill_value="")


def _canonicalize_decision_source_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=DECISION_SOURCE_BASE_COLS)
    src = df.copy()
    out = pd.DataFrame(index=src.index)
    code_col = _first_existing_col(src, CODE_COL_CANDIDATES)
    name_col = _first_existing_col(src, NAME_COL_CANDIDATES)
    board_col = _first_existing_col(src, BOARD_COL_CANDIDATES)
    out["rank"] = _normalize_rank(src)
    out["ts_code"] = src[code_col] if code_col else ""
    out["name"] = src[name_col] if name_col else ""
    out["board"] = src[board_col] if board_col else ""
    prob_series = src["Probability"] if "Probability" in src.columns else (
        src["prob_final"] if "prob_final" in src.columns else (src["prob_ml"] if "prob_ml" in src.columns else "")
    )
    out["prob"] = pd.to_numeric(prob_series, errors="coerce")
    out["StrengthScore"] = pd.to_numeric(
        src["StrengthScore"] if "StrengthScore" in src.columns else (src["强度得分"] if "强度得分" in src.columns else ""),
        errors="coerce",
    )
    out["ThemeBoost"] = pd.to_numeric(
        src["ThemeBoost"] if "ThemeBoost" in src.columns else (src["题材加成"] if "题材加成" in src.columns else ""),
        errors="coerce",
    )
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    return out.reindex(columns=DECISION_SOURCE_BASE_COLS, fill_value="")


def _enrich_prediction_with_meta(df: Optional[pd.DataFrame], trade_date: str, verify_date: str, run_meta: Dict[str, str]) -> pd.DataFrame:
    base = _canonicalize_prediction_frame(df)
    out = base.copy()
    out.insert(0, "trade_date", trade_date)
    out.insert(1, "verify_date", verify_date)
    out["run_id"] = run_meta["run_id"]
    out["run_attempt"] = run_meta["run_attempt"]
    out["commit_sha"] = run_meta["commit_sha"]
    out["generated_at_utc"] = run_meta["generated_at_utc"]
    return out.reindex(columns=V3_PRED_COLS, fill_value="")


def _enrich_decision_source_with_meta(df: Optional[pd.DataFrame], trade_date: str, verify_date: str, run_meta: Dict[str, str]) -> pd.DataFrame:
    base = _canonicalize_decision_source_frame(df)
    out = base.copy()
    out.insert(0, "trade_date", trade_date)
    out.insert(1, "verify_date", verify_date)
    out["run_id"] = run_meta["run_id"]
    out["run_attempt"] = run_meta["run_attempt"]
    out["commit_sha"] = run_meta["commit_sha"]
    out["generated_at_utc"] = run_meta["generated_at_utc"]
    return out.reindex(columns=DECISION_SOURCE_COLS, fill_value="")


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
    out["Probability"] = src["Probability"] if "Probability" in src.columns else (
        src["prob_final"] if "prob_final" in src.columns else (src["prob_ml"] if "prob_ml" in src.columns else "")
    )
    out["_prob_src"] = src["_prob_src"] if "_prob_src" in src.columns else (src["prob_src"] if "prob_src" in src.columns else "")
    out["StrengthScore"] = src["StrengthScore"] if "StrengthScore" in src.columns else ""
    out["ThemeBoost"] = src["ThemeBoost"] if "ThemeBoost" in src.columns else ""
    out["seal_amount"] = _extract_numeric(src, SEAL_AMOUNT_COL_CANDIDATES)
    out["open_times"] = _extract_numeric(src, OPEN_TIMES_COL_CANDIDATES)
    out["turnover_rate"] = _extract_numeric(src, TURNOVER_COL_CANDIDATES)
    out["prob_lr"] = src["prob_lr"] if "prob_lr" in src.columns else ""
    out["prob_lgbm"] = src["prob_lgbm"] if "prob_lgbm" in src.columns else ""
    out["prob_rule"] = src["prob_rule"] if "prob_rule" in src.columns else ""
    for col in [
        "final_score_base",
        "final_score_v2",
        "strength_plus_score",
        "intraday_quality_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "open_board_count",
        "auction_strength_score",
        "auction_real_volume_score",
        "seal_stability_score",
        "intraday_available",
        "auction_available",
        "risk_level",
        "risk_tags",
    ]:
        out[col] = src[col] if col in src.columns else ""

    for col in [
        "is_sample_mature", "mature_reason", "label_delay_flag", "y_limit_hit", "y_next_ret",
        "learnable_flag", "reject_reason", "sample_quality_grade", "batch_quality_score",
        "gate_version", "label_version",
    ]:
        out[col] = src[col] if col in src.columns else ""

    out["verify_date"] = src["verify_date"] if "verify_date" in src.columns else verify_date
    out["close"] = _extract_numeric(src, CLOSE_COL_CANDIDATES)
    if out["close"].isna().all():
        close_map = _build_close_map_from_ctx(ctx)
        if close_map:
            out["close"] = [close_map.get(_safe_str(v), pd.NA) for v in out["ts_code"]]

    for col in [
        "Probability", "StrengthScore", "ThemeBoost", "seal_amount", "open_times", "turnover_rate",
        "prob_lr", "prob_lgbm", "prob_rule", "is_sample_mature", "label_delay_flag",
        "y_limit_hit", "y_next_ret", "learnable_flag", "batch_quality_score", "close",
        "final_score_base", "final_score_v2", "strength_plus_score", "intraday_quality_score",
        "intraday_risk_score", "late_withdraw_score", "reseal_score", "open_board_count",
        "auction_strength_score", "auction_real_volume_score", "seal_stability_score",
        "intraday_available", "auction_available",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop(columns=[c for c in FEATURE_HISTORY_DROP_COLS if c in out.columns], errors="ignore")
    out = out.reindex(columns=FEATURE_HISTORY_COLS, fill_value="")
    out["trade_date"] = out["trade_date"].map(_clean_date_value)
    out["verify_date"] = out["verify_date"].map(_clean_date_value)
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out = out[out["ts_code"] != ""].copy()
    return out.drop_duplicates(subset=FEATURE_HISTORY_REQUIRED_KEYS, keep="last").reset_index(drop=True)


def _normalize_existing_feature_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_HISTORY_COLS)
    d = df.copy()
    if "Probability" not in d.columns:
        d["Probability"] = d["prob_final"] if "prob_final" in d.columns else (d["prob_ml"] if "prob_ml" in d.columns else "")
    if "_prob_src" not in d.columns:
        d["_prob_src"] = d["prob_src"] if "prob_src" in d.columns else ""
    for col in FEATURE_HISTORY_COLS:
        if col not in d.columns:
            d[col] = ""
    d = d.drop(columns=[c for c in FEATURE_HISTORY_DROP_COLS if c in d.columns], errors="ignore")
    d = d.reindex(columns=FEATURE_HISTORY_COLS, fill_value="")
    d["trade_date"] = d["trade_date"].map(_clean_date_value)
    d["verify_date"] = d["verify_date"].map(_clean_date_value)
    d["ts_code"] = d["ts_code"].astype(str).str.strip()
    d = d[(d["trade_date"] != "") & (d["ts_code"] != "")].copy()
    return d.drop_duplicates(subset=FEATURE_HISTORY_REQUIRED_KEYS, keep="last").reset_index(drop=True)


def _merge_feature_history(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    existing_df = _normalize_existing_feature_history(existing_df)
    new_df = _normalize_existing_feature_history(new_df)
    if existing_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = existing_df.copy()
    else:
        existing_map = {(str(r["trade_date"]), str(r["ts_code"])): r.to_dict() for _, r in existing_df.iterrows()}
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
    return merged.drop_duplicates(subset=FEATURE_HISTORY_REQUIRED_KEYS, keep="last").reset_index(drop=True)


def _format_probability(x: Any) -> str:
    try:
        v = float(x)
        if pd.isna(v):
            return ""
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return ""


def _format_score(x: Any) -> str:
    try:
        v = float(x)
        if pd.isna(v):
            return ""
        return f"{v:.4f}".rstrip("0").rstrip(".")
    except Exception:
        return ""


def _format_pct(x: Any) -> str:
    try:
        v = float(x)
        if pd.isna(v):
            return ""
        return f"{v:.1%}"
    except Exception:
        return ""


def _format_ret_pct(x: Any) -> str:
    try:
        v = float(x)
        if pd.isna(v):
            return ""
        return f"{v * 100:.2f}%"
    except Exception:
        return ""


def _rename_human(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "rank": "排名",
        "ts_code": "代码",
        "name": "股票",
        "Probability": "涨停概率",
        "_prob_src": "概率来源",
        "StrengthScore": "强度得分",
        "strength_plus_score": "强度分",
        "ThemeBoost": "题材加成",
        "final_score_v2": "最终分",
        "final_score_base": "原始分",
        "intraday_quality_score": "分时质量",
        "intraday_risk_score": "分时风险",
        "late_withdraw_score": "尾盘风险",
        "reseal_score": "回封分",
        "open_board_count": "炸板数",
        "auction_strength_score": "竞价强度",
        "risk_level": "风险级别",
        "risk_tags": "风险标签",
        "intraday_data_status": "分时状态",
        "board": "板块",
        "trade_date": "预测日",
        "verify_date": "验证日",
        "命中": "是否命中",
        "next_ret": "次日涨跌幅",
        "topn": "TopN",
        "hit": "命中数",
        "hit_rate": "命中率",
        "limit_count": "当日涨停家数",
        "avg_ret": "平均涨跌幅",
        "median_ret": "中位涨跌幅",
    }
    d = df.copy()
    use = {c: mapping[c] for c in d.columns if c in mapping}
    return d.rename(columns=use)


def _df_to_md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    if df is None or df.empty:
        return ""
    d = df.copy()
    if cols is not None:
        use_cols = [c for c in cols if c in d.columns]
        if use_cols:
            d = d[use_cols].copy()
    d = _rename_human(d)
    try:
        return d.to_markdown(index=False)
    except Exception:
        x = d.fillna("")
        headers = list(x.columns)

        def esc(v: Any) -> str:
            return str(v).replace("\n", " ").replace("\r", " ").replace("|", "\\|")

        lines = ["| " + " | ".join(esc(h) for h in headers) + " |"]
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for _, row in x.iterrows():
            lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
        return "\n".join(lines)


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
    return _to_df(payload.get("topN") or [])


def _limit_hit_code_sets(limit_df: pd.DataFrame) -> Tuple[set, set]:
    _, ts_set, c6_set = _build_code_sets(limit_df, CODE_COL_CANDIDATES)
    return ts_set, c6_set


def _topn_to_hit_df(
    topn_df: Optional[pd.DataFrame],
    limit_df: pd.DataFrame,
    verify_daily_df: Optional[pd.DataFrame] = None,
    verify_date: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pred = _canonicalize_prediction_frame(topn_df)
    metrics = {"hit_count": 0, "top_count": 0, "limit_count": _count_limitups(limit_df), "hit_rate": "", "avg_ret": "", "median_ret": ""}
    if pred.empty:
        return pd.DataFrame(), metrics
    limit_ts, limit_c6 = _limit_hit_code_sets(limit_df)
    close_col = _first_existing_col(verify_daily_df, CLOSE_COL_CANDIDATES) if verify_daily_df is not None and not verify_daily_df.empty else None
    code_col = _first_existing_col(verify_daily_df, CODE_COL_CANDIDATES) if verify_daily_df is not None and not verify_daily_df.empty else None
    next_close_map: Dict[str, float] = {}
    if code_col and close_col:
        for _, row in verify_daily_df[[code_col, close_col]].dropna(subset=[code_col]).iterrows():
            ts, c6 = _norm_code(row[code_col])
            val = pd.to_numeric(row[close_col], errors="coerce")
            if pd.notna(val):
                if ts:
                    next_close_map[ts] = float(val)
                if c6:
                    next_close_map[c6] = float(val)

    hits: List[str] = []
    rets: List[Any] = []
    hit_count = 0
    for _, row in pred.iterrows():
        ts, c6 = _norm_code(row["ts_code"])
        ok = (ts in limit_ts) or (c6 in limit_c6)
        hits.append("是" if ok else "否")
        if ok:
            hit_count += 1
        tclose = pd.to_numeric(row.get("close"), errors="coerce")
        nclose = next_close_map.get(ts) if ts else next_close_map.get(c6)
        if pd.notna(tclose) and nclose is not None and float(tclose) != 0:
            rets.append(float(nclose) / float(tclose) - 1.0)
        else:
            rets.append(pd.NA)

    out = pred.copy()
    out.insert(0, "trade_date", "")
    out.insert(1, "verify_date", verify_date)
    out["命中"] = hits
    out["next_ret"] = rets
    out = out.sort_values(by=["命中", "next_ret"], ascending=[False, False], na_position="last").reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    metrics["hit_count"] = hit_count
    metrics["top_count"] = int(len(out))
    metrics["hit_rate"] = f"{round(hit_count * 100.0 / float(len(out)))}%" if len(out) else ""
    ret_s = pd.to_numeric(pd.Series(rets), errors="coerce")
    if ret_s.notna().any():
        metrics["avg_ret"] = _format_ret_pct(ret_s.mean())
        metrics["median_ret"] = _format_ret_pct(ret_s.median())
    return out, metrics


def _load_step7_hit_history(learning_dir: Path, max_days: int = 10) -> pd.DataFrame:
    p = learning_dir / "step7_hit_rate_history.csv"
    df = _read_csv_guess(p)
    if df.empty:
        return pd.DataFrame()
    for c in ["trade_date", "verify_date", "topn", "hit", "hit_rate", "note"]:
        if c not in df.columns:
            df[c] = ""
    df["trade_date"] = df["trade_date"].astype(str)
    df["verify_date"] = df["verify_date"].astype(str)
    df["topn"] = pd.to_numeric(df["topn"], errors="coerce")
    df["hit"] = pd.to_numeric(df["hit"], errors="coerce")
    df["hit_rate"] = pd.to_numeric(df["hit_rate"], errors="coerce")
    df = df[df["hit_rate"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("trade_date").tail(max_days).reset_index(drop=True)


def _build_recent_perf_df(hit_hist: pd.DataFrame, settings, ctx) -> pd.DataFrame:
    if hit_hist is None or hit_hist.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, row in hit_hist.iterrows():
        td = _clean_date_value(row.get("trade_date"))
        vd = _clean_date_value(row.get("verify_date"))
        limit_df = _load_limit_df(settings, {}, vd) if vd else pd.DataFrame()
        hit_v = pd.to_numeric(row.get("hit"), errors="coerce")
        limit_count = _count_limitups(limit_df)
        rows.append({
            "trade_date": td,
            "verify_date": vd,
            "hit": "" if pd.isna(hit_v) else int(hit_v),
            "hit_rate": _format_pct(pd.to_numeric(row.get("hit_rate"), errors="coerce")),
            "limit_count": "" if limit_count <= 0 else int(limit_count),
        })
    return pd.DataFrame(rows)


def _build_recent_summary_df(hit_hist: pd.DataFrame) -> pd.DataFrame:
    if hit_hist is None or hit_hist.empty:
        return pd.DataFrame()
    hit = pd.to_numeric(hit_hist["hit"], errors="coerce").fillna(0)
    topn = pd.to_numeric(hit_hist["topn"], errors="coerce").fillna(0)
    hr = pd.to_numeric(hit_hist["hit_rate"], errors="coerce")
    rows = [
        {"指标": "近10日总命中数", "数值": int(hit.sum())},
        {"指标": "近10日总预测数", "数值": int(topn.sum())},
        {"指标": "近10日平均命中率", "数值": _format_pct(hr.mean()) if hr.notna().any() else ""},
        {"指标": "近10日最佳单日命中率", "数值": _format_pct(hr.max()) if hr.notna().any() else ""},
        {"指标": "近10日最差单日命中率", "数值": _format_pct(hr.min()) if hr.notna().any() else ""},
    ]
    return pd.DataFrame(rows)


def _standardize_candidate_pool(full_df: Optional[pd.DataFrame], topn_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    full_v3 = _canonicalize_prediction_frame(full_df)
    if full_v3.empty:
        return pd.DataFrame()
    top_codes = set(_canonicalize_prediction_frame(topn_df)["ts_code"].astype(str).tolist()) if topn_df is not None else set()
    out = full_v3[~full_v3["ts_code"].astype(str).isin(top_codes)].copy()
    out = out.sort_values(by=["Probability", "StrengthScore"], ascending=[False, False], na_position="last").reset_index(drop=True)
    out["rank"] = range(11, len(out) + 11)
    return out


def _write_feature_history(learning_dir: Path, batch_df: pd.DataFrame) -> pd.DataFrame:
    history_path = learning_dir / "feature_history.csv"
    existing_df = _read_csv_guess(history_path)
    merged_df = _merge_feature_history(existing_df, batch_df)
    _write_csv_overwrite(merged_df, history_path)
    return merged_df


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn) -> None:
    outdir = Path(getattr(getattr(settings, "io", None), "outputs_dir", None) or "outputs")
    _ensure_dir(outdir)

    trade_date = _clean_date_value(trade_date)
    run_meta = _get_run_meta()
    learning_dir = outdir / "learning"
    warehouse_dir = outdir / "_warehouse" / "pred_top10"
    decisio_dir = outdir / "decisio"
    _ensure_dir(learning_dir)
    _ensure_dir(warehouse_dir)
    _ensure_dir(decisio_dir)

    topn_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None
    if isinstance(topn, dict):
        raw_topn = None
        for k in ["topN", "topn", "TopN", "top"]:
            if k in topn and topn.get(k) is not None:
                raw_topn = topn.get(k)
                break
        topn_df = _to_df(raw_topn)
        full_df = _to_df(topn.get("full")) if "full" in topn else None
    else:
        topn_df = _to_df(topn)

    calendar = _list_trade_dates(settings)
    prev_td, next_td = _prev_next_trade_date_with_fallback(calendar, trade_date)

    limit_df_current = _load_limit_df(settings, ctx, trade_date)
    daily_df_current = _load_daily_df(settings, ctx, trade_date)
    _, metrics_same_day = _topn_to_hit_df(topn_df, limit_df_current, daily_df_current, trade_date)

    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "verify_date": next_td,
        "gate": gate,
        "topN": [] if topn_df is None else topn_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics_same_day,
        "run_meta": run_meta,
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    _write_text_overwrite(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    try:
        intraday_debug_path = outdir / f"debug_intraday_{trade_date}.json"
        src_for_debug = full_df if full_df is not None and not full_df.empty else topn_df
        dbg_df = _to_df(src_for_debug)
        top_dbg = _to_df(topn_df)
        ctx_debug = ctx.get("debug", {}) if isinstance(ctx, dict) else {}
        intraday_input = ctx_debug.get("intraday_input", {}) if isinstance(ctx_debug, dict) else {}
        debug_intraday = {
            "intraday_enabled": bool(getattr(getattr(settings, "intraday", None), "enabled", True)),
            "intraday_rows": int(intraday_input.get("intraday_features_rows", 0) or 0),
            "auction_rows": int(intraday_input.get("stk_auction_rows", 0) or 0),
            "candidate_rows": int(len(dbg_df)) if dbg_df is not None else 0,
            "matched_intraday_rows": int((pd.to_numeric(dbg_df.get("intraday_available"), errors="coerce").fillna(0) > 0).sum()) if dbg_df is not None and not dbg_df.empty and "intraday_available" in dbg_df.columns else 0,
            "matched_auction_rows": int((pd.to_numeric(dbg_df.get("auction_available"), errors="coerce").fillna(0) > 0).sum()) if dbg_df is not None and not dbg_df.empty and "auction_available" in dbg_df.columns else 0,
            "topn_intraday_available_ratio": float((pd.to_numeric(top_dbg.get("intraday_available"), errors="coerce").fillna(0) > 0).mean()) if top_dbg is not None and not top_dbg.empty and "intraday_available" in top_dbg.columns else 0.0,
            "topn_hard_risk_count": int((pd.to_numeric(top_dbg.get("intraday_hard_risk_flag"), errors="coerce").fillna(0) > 0).sum()) if top_dbg is not None and not top_dbg.empty and "intraday_hard_risk_flag" in top_dbg.columns else 0,
            "fallback_used": bool(int(intraday_input.get("intraday_features_rows", 0) or 0) == 0 or int(intraday_input.get("stk_auction_rows", 0) or 0) == 0),
        }
        _write_text_overwrite(intraday_debug_path, json.dumps(debug_intraday, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] debug_intraday write failed: {e}")

    topn_v3 = _canonicalize_prediction_frame(topn_df)
    candidate_pool_df = _standardize_candidate_pool(full_df, topn_df)

    prev_topn_df = _load_json_topn(outdir, prev_td)
    prev_verify_daily = _load_daily_df(settings, ctx, trade_date)
    prev_hit_df, _ = _topn_to_hit_df(prev_topn_df, limit_df_current, prev_verify_daily, trade_date)
    if not prev_hit_df.empty:
        prev_hit_df["trade_date"] = prev_td
        prev_hit_df["verify_date"] = trade_date
        for col in ["Probability", "StrengthScore", "ThemeBoost"]:
            if col in prev_hit_df.columns:
                prev_hit_df[col] = prev_hit_df[col].map(_format_score if col != "Probability" else _format_probability)
        prev_hit_df["next_ret"] = prev_hit_df["next_ret"].map(_format_ret_pct)
        prev_hit_df["rank"] = range(1, len(prev_hit_df) + 1)

    step7_hist = _load_step7_hit_history(learning_dir, max_days=10)
    perf_df = _build_recent_perf_df(step7_hist, settings, ctx)
    perf_summary_df = _build_recent_summary_df(step7_hist)

    if not topn_v3.empty:
        for col in [
            "Probability", "StrengthScore", "ThemeBoost", "final_score_v2", "final_score_base",
            "strength_plus_score", "intraday_quality_score", "intraday_risk_score",
            "late_withdraw_score", "reseal_score", "auction_strength_score",
        ]:
            if col in topn_v3.columns:
                topn_v3[col] = topn_v3[col].map(_format_score if col != "Probability" else _format_probability)
    if not candidate_pool_df.empty:
        for col in [
            "Probability", "StrengthScore", "ThemeBoost", "final_score_v2", "final_score_base",
            "strength_plus_score", "intraday_quality_score", "intraday_risk_score",
            "late_withdraw_score", "reseal_score", "auction_strength_score",
        ]:
            if col in candidate_pool_df.columns:
                candidate_pool_df[col] = candidate_pool_df[col].map(_format_score if col != "Probability" else _format_probability)

    md_lines: List[str] = [f"# {trade_date} 预测报告\n"]
    md_lines.append(
        "本次 Top10 排名已接入 a-share-top3-data 上游分时增强数据。"
        "新增数据包括集合竞价、涨停路径、炸板回封、尾盘撤退等。"
        "该模块主要用于识别 D 日涨停质量和隔夜风险。"
        "当分时数据缺失时，系统按中性值降级，不把缺失视为高风险。\n"
    )
    md_lines.append(f"## {trade_date} 预测：{next_td} 涨停 Top10（按涨停概率降序）\n")
    if topn_v3.empty:
        reason = _safe_str(gate.get("reason") if isinstance(gate, dict) else "")
        md_lines.append(f"⚠️ Gate 未通过，Top10 为空。{reason}\n")
    else:
        md_lines.append(_df_to_md_table(topn_v3, cols=[
            "rank", "ts_code", "name", "final_score_v2", "final_score_base", "Probability",
            "strength_plus_score", "ThemeBoost", "intraday_quality_score", "intraday_risk_score",
            "late_withdraw_score", "reseal_score", "open_board_count", "auction_strength_score",
            "risk_level", "risk_tags", "intraday_data_status",
        ]))
        md_lines.append("")

    md_lines.append(f"## {trade_date} 候选池补充表（Top10 之外，按涨停概率降序）\n")
    if candidate_pool_df.empty:
        md_lines.append("（无 Top10 之外的候选样本）\n")
    else:
        md_lines.append(_df_to_md_table(candidate_pool_df, cols=[
            "rank", "ts_code", "name", "final_score_v2", "Probability", "strength_plus_score",
            "ThemeBoost", "intraday_quality_score", "intraday_risk_score", "risk_level",
            "risk_tags", "intraday_data_status",
        ]))
        md_lines.append("")

    md_lines.append(f"## {prev_td} 预测在 {trade_date} 收盘后的命中情况\n")
    if prev_hit_df.empty:
        md_lines.append("（未找到上一交易日预测文件或上一交易日 Top10 为空）\n")
    else:
        md_lines.append(_df_to_md_table(
            prev_hit_df,
            cols=["rank", "trade_date", "verify_date", "ts_code", "name", "Probability", "_prob_src", "StrengthScore", "ThemeBoost", "board", "命中", "next_ret"],
        ))
        md_lines.append("")

    if not perf_df.empty:
        md_lines.append("## 近10日 Top10 绩效\n")
        md_lines.append(_df_to_md_table(perf_df, cols=["trade_date", "verify_date", "hit", "hit_rate", "limit_count"]))
        md_lines.append("")
        if not perf_summary_df.empty:
            md_lines.append("### 近10日整体统计\n")
            md_lines.append(_df_to_md_table(perf_summary_df))
            md_lines.append("")

    md_text = "\n".join(md_lines)
    md_path = outdir / f"predict_top10_{trade_date}.md"
    _write_text_overwrite(md_path, md_text, encoding="utf-8")
    _write_text_overwrite(outdir / "latest.md", md_text, encoding="utf-8")

    topn_out = _enrich_prediction_with_meta(topn_df, trade_date, next_td, run_meta)
    full_out = _enrich_decision_source_with_meta(full_df, trade_date, next_td, run_meta)

    _write_csv_overwrite(topn_out, learning_dir / f"pred_top10_{trade_date}.csv")
    _write_csv_overwrite(topn_out, learning_dir / "pred_top10_latest.csv")
    _write_csv_once(topn_out, warehouse_dir / f"pred_top10_{trade_date}_{run_meta['run_id']}.csv")

    _write_csv_overwrite(full_out, decisio_dir / f"pred_decisio_{trade_date}.csv")
    _write_csv_overwrite(full_out, decisio_dir / "pred_decisio_latest.csv")

    feature_source = full_df if full_df is not None and not full_df.empty else topn_df
    feature_batch = _canonicalize_feature_history_batch(
        feature_source,
        trade_date=trade_date,
        verify_date=next_td,
        run_time_utc=run_meta["generated_at_utc"],
        ctx=ctx,
    )
    merged_history = _write_feature_history(learning_dir, feature_batch)

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
