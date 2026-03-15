#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step3: Strength Score — Top10 V3

定位：
- Step3 不再只是“算一个分”，而是 V3 的盘口强弱特征聚合器。
- 负责把候选池样本补齐为可下游消费的强度字段契约。
- 明确区分：
    1) 原始强度聚合值：limit_strength_raw
    2) 最终标准化得分：StrengthScore
- 严格执行 V3 缺失策略：
    - turnover_rate 缺失：保留为空，不伪造 0
    - seal_amount 缺失：保留为空，不伪造 0
    - open_times 缺失：保留为空，不伪造 0

核心输出：
- StrengthScore
- limit_strength_raw
- strength_feature_count
- strength_quality_flag
- strength_missing_fields

并尽量透传：
- turnover_rate
- open_times
- seal_amount
- limit_type
- up_limit
- down_limit
- is_limit_up_pool
- first_limit_time
- last_limit_time
- limit_open_count

旁路输出：
- outputs/step3_strength_YYYYMMDD.csv
- outputs/debug_step3_report.md
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Basic utils
# =========================================================

def _coerce_df(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, Mapping):
        for k in (
            "candidates",
            "candidate",
            "df",
            "data",
            "result",
            "pool",
            "step2",
            "step2_candidates",
        ):
            v = obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
    return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if str(name).lower() in lower_map:
            return lower_map[str(name).lower()]
    return None


def _as_str_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: pd.NA if isinstance(x, str) and x == "" else x)
    return s


def _to_float_nullable(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    """
    保留缺失为 NaN，不伪造 0。
    """
    if (col is None) or (col not in df.columns):
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _to_float_fill(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    """
    用于非契约关键字段的内部计算兜底。
    """
    s = _to_float_nullable(df, col)
    return s.fillna(default).astype("float64")


def _clip01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.clip(0.0, 1.0)


def _robust_rank01(s: pd.Series, ascending: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    valid = s.notna().sum()
    if valid == 0:
        return pd.Series([0.0] * len(s), index=s.index, dtype="float64")
    return s.rank(pct=True, method="average", ascending=ascending).fillna(0.0).astype("float64")


def _log1p_safe(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = s.clip(0.0, float("inf"))
    return pd.Series(np.log1p(s.to_numpy(dtype="float64")), index=s.index, dtype="float64")


def _resolve_trade_date(s=None) -> str:
    td = os.getenv("TRADE_DATE", "").strip()
    if td:
        return td
    if s is not None:
        for k in ("trade_date", "TRADE_DATE"):
            if hasattr(s, k):
                v = str(getattr(s, k) or "").strip()
                if v:
                    return v
    return datetime.now().strftime("%Y%m%d")


def _normalize_ts_code(df: pd.DataFrame, col: str = "ts_code") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.strip()
    return df


def _pick_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return _first_existing_col(df, names)


def _safe_bool_from_flag(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    return (s.fillna(0.0) > 0.5).astype("float64")


# =========================================================
# Snapshot locating
# =========================================================

def _candidate_snapshot_dirs(trade_date: str) -> Sequence[Path]:
    y = trade_date[:4]
    return [
        Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / trade_date,
        Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / trade_date,
        Path("data_repo") / "snapshots" / trade_date,
        Path("snapshots") / trade_date,
    ]


def _locate_snapshot_dir(
    trade_date: str,
    s=None,
    ctx: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    if ctx and isinstance(ctx, dict):
        p = ctx.get("snapshot_dir")
        if p:
            pp = Path(str(p))
            if pp.exists():
                return pp

    if s is not None and hasattr(s, "snapshot_dir"):
        try:
            pp = Path(str(s.snapshot_dir(trade_date)))
            if pp.exists():
                return pp
        except Exception:
            pass

    for p in _candidate_snapshot_dirs(trade_date):
        if p.exists():
            return p
    return None


def _read_csv(p: Path) -> pd.DataFrame:
    if not p or not p.exists():
        return pd.DataFrame()
    for kwargs in (
        {},
        {"encoding": "utf-8", "errors": "ignore"},
        {"encoding": "gbk", "errors": "ignore"},
    ):
        try:
            return pd.read_csv(p, **kwargs)
        except Exception:
            continue
    return pd.DataFrame()


# =========================================================
# Candidate loading / identity
# =========================================================

def _load_candidates_fallback(trade_date: str) -> pd.DataFrame:
    p = Path("outputs") / f"step2_candidates_{trade_date}.csv"
    return _read_csv(p) if p.exists() else pd.DataFrame()


def _ensure_identity(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    ts_col = _first_existing_col(
        out,
        ["ts_code", "TS_CODE", "code", "symbol", "ticker", "证券代码", "代码"],
    ) or "ts_code"
    out["ts_code"] = _as_str_series(out, ts_col)

    name_col = _first_existing_col(
        out,
        ["name", "NAME", "stock_name", "名称", "股票", "证券名称"],
    )
    out["name"] = _as_str_series(out, name_col) if name_col else ""

    ind_col = _first_existing_col(
        out,
        ["industry", "行业", "板块", "industry_name", "所属行业"],
    )
    out["industry"] = _as_str_series(out, ind_col) if ind_col else ""

    td_col = _first_existing_col(out, ["trade_date", "TRADE_DATE", "日期"])
    if td_col:
        out["trade_date"] = out[td_col].astype(str)
    elif "trade_date" not in out.columns:
        out["trade_date"] = ""

    return out


# =========================================================
# Snapshot feature extraction
# =========================================================

def _norm_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    td_col = _pick_col(df, ["trade_date", "TRADE_DATE", "日期"])
    out = df.copy()
    if td_col and td_col != "trade_date":
        out = out.rename(columns={td_col: "trade_date"})
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype(str)
    return out


def _extract_numeric_by_day(
    df: pd.DataFrame,
    out_col: str,
    value_candidates: Sequence[str],
    agg: str = "sum",
) -> pd.DataFrame:
    if df is None or df.empty or "ts_code" not in df.columns or "trade_date" not in df.columns:
        return pd.DataFrame()

    value_col = _pick_col(df, value_candidates)
    if not value_col:
        return pd.DataFrame()

    x = df[["ts_code", "trade_date", value_col]].copy()
    x = x.rename(columns={value_col: out_col})
    x[out_col] = pd.to_numeric(x[out_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if agg == "max":
        x = x.groupby(["ts_code", "trade_date"], as_index=False)[out_col].max()
    elif agg == "min":
        x = x.groupby(["ts_code", "trade_date"], as_index=False)[out_col].min()
    else:
        x = x.groupby(["ts_code", "trade_date"], as_index=False)[out_col].sum()
    return x


def _extract_text_by_day(
    df: pd.DataFrame,
    out_col: str,
    value_candidates: Sequence[str],
) -> pd.DataFrame:
    if df is None or df.empty or "ts_code" not in df.columns or "trade_date" not in df.columns:
        return pd.DataFrame()

    value_col = _pick_col(df, value_candidates)
    if not value_col:
        return pd.DataFrame()

    x = df[["ts_code", "trade_date", value_col]].copy()
    x = x.rename(columns={value_col: out_col})
    x[out_col] = x[out_col].astype("object")
    x[out_col] = x[out_col].map(lambda v: v.strip() if isinstance(v, str) else v)
    x[out_col] = x[out_col].map(lambda v: pd.NA if isinstance(v, str) and v == "" else v)
    x = x.groupby(["ts_code", "trade_date"], as_index=False)[out_col].first()
    return x


def _join_snap_features(cand: pd.DataFrame, snap_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    debug: Dict[str, Any] = {"snapshot_dir": str(snap_dir), "files": {}}

    daily = _norm_trade_date(_read_csv(snap_dir / "daily.csv"))
    daily_basic = _norm_trade_date(_read_csv(snap_dir / "daily_basic.csv"))
    top_list = _norm_trade_date(_read_csv(snap_dir / "top_list.csv"))
    hsgt = _norm_trade_date(_read_csv(snap_dir / "moneyflow_hsgt.csv"))
    limit_list = _norm_trade_date(_read_csv(snap_dir / "limit_list_d.csv"))
    limit_break = _norm_trade_date(_read_csv(snap_dir / "limit_break_d.csv"))
    stk_limit = _norm_trade_date(_read_csv(snap_dir / "stk_limit.csv"))

    for name, df in (
        ("daily.csv", daily),
        ("daily_basic.csv", daily_basic),
        ("top_list.csv", top_list),
        ("moneyflow_hsgt.csv", hsgt),
        ("limit_list_d.csv", limit_list),
        ("limit_break_d.csv", limit_break),
        ("stk_limit.csv", stk_limit),
    ):
        debug["files"][name] = int(len(df))

    out = cand.copy()
    out = _normalize_ts_code(out, "ts_code")

    for df in (daily, daily_basic, top_list, hsgt, limit_list, limit_break, stk_limit):
        if not df.empty and "ts_code" in df.columns:
            _normalize_ts_code(df, "ts_code")

    # daily
    if not daily.empty and {"ts_code", "trade_date"}.issubset(daily.columns):
        pct_col = _pick_col(daily, ["pct_chg", "pct_change", "change_pct", "涨跌幅"])
        amt_col = _pick_col(daily, ["amount", "成交额", "turnover_amount", "amt"])
        vol_col = _pick_col(daily, ["vol", "成交量", "volume"])
        close_col = _pick_col(daily, ["close", "收盘价"])

        keep = ["ts_code", "trade_date"] + [c for c in [pct_col, amt_col, vol_col, close_col] if c]
        d = daily[keep].copy()
        rename_map = {}
        if pct_col and pct_col != "pct_chg":
            rename_map[pct_col] = "pct_chg"
        if amt_col and amt_col != "amount":
            rename_map[amt_col] = "amount"
        if vol_col and vol_col != "vol":
            rename_map[vol_col] = "vol"
        if close_col and close_col != "close":
            rename_map[close_col] = "close"
        if rename_map:
            d = d.rename(columns=rename_map)
        out = out.merge(d, on=["ts_code", "trade_date"], how="left")

    # daily_basic
    if not daily_basic.empty and {"ts_code", "trade_date"}.issubset(daily_basic.columns):
        trf_col = _pick_col(daily_basic, ["turnover_rate_f", "turnover_f"])
        tr_col = _pick_col(daily_basic, ["turnover_rate", "turn_rate", "换手率"])
        cmv_col = _pick_col(daily_basic, ["circ_mv", "float_mv", "流通市值"])
        tmv_col = _pick_col(daily_basic, ["total_mv", "总市值"])
        vr_col = _pick_col(daily_basic, ["volume_ratio", "量比"])

        keep = ["ts_code", "trade_date"] + [c for c in [trf_col, tr_col, cmv_col, tmv_col, vr_col] if c]
        db = daily_basic[keep].copy()
        rename_map = {}
        if trf_col and trf_col != "turnover_rate_f":
            rename_map[trf_col] = "turnover_rate_f"
        if tr_col and tr_col != "turnover_rate":
            rename_map[tr_col] = "turnover_rate"
        if cmv_col and cmv_col != "circ_mv":
            rename_map[cmv_col] = "circ_mv"
        if tmv_col and tmv_col != "total_mv":
            rename_map[tmv_col] = "total_mv"
        if vr_col and vr_col != "volume_ratio":
            rename_map[vr_col] = "volume_ratio"
        if rename_map:
            db = db.rename(columns=rename_map)
        out = out.merge(db, on=["ts_code", "trade_date"], how="left")

    # 龙虎榜 / 北向
    def extract_net_amount(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df is None or df.empty or not {"ts_code", "trade_date"}.issubset(df.columns):
            return pd.DataFrame()
        cand_cols = [c for c in df.columns if "net" in str(c).lower()]
        net_col = _pick_col(df, ["net_amount", "net_amt", "net_buy", "net"]) or (cand_cols[0] if cand_cols else None)
        if not net_col:
            return pd.DataFrame()
        x = df[["ts_code", "trade_date", net_col]].copy()
        x = x.rename(columns={net_col: f"{prefix}_net_amount"})
        x[f"{prefix}_net_amount"] = pd.to_numeric(
            x[f"{prefix}_net_amount"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        x = x.groupby(["ts_code", "trade_date"], as_index=False)[f"{prefix}_net_amount"].sum()
        return x

    tl = extract_net_amount(top_list, "lhb")
    if not tl.empty:
        out = out.merge(tl, on=["ts_code", "trade_date"], how="left")

    hs = extract_net_amount(hsgt, "hsgt")
    if not hs.empty:
        out = out.merge(hs, on=["ts_code", "trade_date"], how="left")

    # limit_list
    if not limit_list.empty and {"ts_code", "trade_date"}.issubset(limit_list.columns):
        ll = limit_list[["ts_code", "trade_date"]].drop_duplicates().copy()
        ll["is_limit_up_pool"] = 1.0
        out = out.merge(ll, on=["ts_code", "trade_date"], how="left")

        for joined in (
            _extract_numeric_by_day(limit_list, "seal_amount", ["seal_amount", "fd_amount", "fund_amount", "封单额", "封单金额", "封单资金"]),
            _extract_text_by_day(limit_list, "limit_type", ["limit_type", "涨停类型"]),
            _extract_numeric_by_day(limit_list, "up_limit", ["up_limit", "涨停价"], agg="max"),
            _extract_numeric_by_day(limit_list, "down_limit", ["down_limit", "跌停价"], agg="max"),
            _extract_text_by_day(limit_list, "first_limit_time", ["first_limit_time", "首次封板时间", "first_time"]),
            _extract_text_by_day(limit_list, "last_limit_time", ["last_limit_time", "最后封板时间", "last_time"]),
        ):
            if not joined.empty:
                out = out.merge(joined, on=["ts_code", "trade_date"], how="left")

    # limit_break
    if not limit_break.empty and {"ts_code", "trade_date"}.issubset(limit_break.columns):
        lb = limit_break[["ts_code", "trade_date"]].drop_duplicates().copy()
        lb["limit_break_flag"] = 1.0
        out = out.merge(lb, on=["ts_code", "trade_date"], how="left")

        for joined in (
            _extract_numeric_by_day(limit_break, "open_times", ["open_times", "open_num", "break_times", "break_count", "炸板次数", "open_cnt", "times"]),
            _extract_numeric_by_day(limit_break, "limit_open_count", ["limit_open_count", "open_times", "open_num", "break_times", "break_count", "炸板次数", "open_cnt", "times"]),
        ):
            if not joined.empty:
                out = out.merge(joined, on=["ts_code", "trade_date"], how="left")

    # stk_limit 作为 up/down limit 回退源
    if not stk_limit.empty and {"ts_code", "trade_date"}.issubset(stk_limit.columns):
        sl_keep = ["ts_code", "trade_date"]
        sl_up = _pick_col(stk_limit, ["up_limit", "涨停价"])
        sl_down = _pick_col(stk_limit, ["down_limit", "跌停价"])
        sl_type = _pick_col(stk_limit, ["limit_type", "涨停类型"])
        for c in (sl_up, sl_down, sl_type):
            if c:
                sl_keep.append(c)

        sl = stk_limit[sl_keep].copy()
        rename_map = {}
        if sl_up and sl_up != "up_limit":
            rename_map[sl_up] = "up_limit_stk"
        if sl_down and sl_down != "down_limit":
            rename_map[sl_down] = "down_limit_stk"
        if sl_type and sl_type != "limit_type":
            rename_map[sl_type] = "limit_type_stk"
        if rename_map:
            sl = sl.rename(columns=rename_map)
        out = out.merge(sl, on=["ts_code", "trade_date"], how="left")

    # 回退 up_limit / down_limit / limit_type
    if "up_limit" not in out.columns and "up_limit_stk" in out.columns:
        out["up_limit"] = out["up_limit_stk"]
    elif "up_limit" in out.columns and "up_limit_stk" in out.columns:
        out["up_limit"] = pd.to_numeric(out["up_limit"], errors="coerce").combine_first(
            pd.to_numeric(out["up_limit_stk"], errors="coerce")
        )

    if "down_limit" not in out.columns and "down_limit_stk" in out.columns:
        out["down_limit"] = out["down_limit_stk"]
    elif "down_limit" in out.columns and "down_limit_stk" in out.columns:
        out["down_limit"] = pd.to_numeric(out["down_limit"], errors="coerce").combine_first(
            pd.to_numeric(out["down_limit_stk"], errors="coerce")
        )

    if "limit_type" not in out.columns and "limit_type_stk" in out.columns:
        out["limit_type"] = out["limit_type_stk"]
    elif "limit_type" in out.columns and "limit_type_stk" in out.columns:
        out["limit_type"] = out["limit_type"].combine_first(out["limit_type_stk"])

    return out, debug


# =========================================================
# Core scoring
# =========================================================

def _build_strength_quality_fields(out: pd.DataFrame) -> pd.DataFrame:
    micro_cols = ["turnover_rate", "open_times", "seal_amount"]
    feature_cols = [
        "turnover_rate",
        "open_times",
        "seal_amount",
        "is_limit_up_pool",
        "limit_type",
        "up_limit",
        "first_limit_time",
        "last_limit_time",
        "limit_open_count",
    ]

    feature_count = pd.Series([0] * len(out), index=out.index, dtype="int64")
    for col in feature_cols:
        if col in out.columns:
            feature_count += out[col].notna().astype("int64")
    out["strength_feature_count"] = feature_count

    missing_fields: List[str] = []
    for idx in out.index:
        miss = []
        for col in micro_cols:
            if col not in out.columns or pd.isna(out.at[idx, col]):
                miss.append(col)
        missing_fields.append("|".join(miss) if miss else "")
    out["strength_missing_fields"] = missing_fields

    micro_nonnull_count = pd.Series([0] * len(out), index=out.index, dtype="int64")
    for col in micro_cols:
        if col in out.columns:
            micro_nonnull_count += out[col].notna().astype("int64")

    quality = pd.Series(["D"] * len(out), index=out.index, dtype="object")
    quality.loc[micro_nonnull_count >= 2] = "A"
    quality.loc[micro_nonnull_count == 1] = "B"
    quality.loc[micro_nonnull_count == 0] = "C"
    out["strength_quality_flag"] = quality

    return out


def calc_strength_score(
    df: Any,
    trade_date: str,
    s=None,
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    debug: Dict[str, Any] = {"trade_date": trade_date, "step": "step3_strength_score_v3"}

    cand = _coerce_df(df)
    if cand.empty:
        cand = _load_candidates_fallback(trade_date)

    cand = _ensure_identity(cand)

    if cand.empty:
        return cand, {"error": "Step3: candidates 为空，无法计算。", "trade_date": trade_date}

    if "trade_date" not in cand.columns:
        cand["trade_date"] = trade_date
    cand["trade_date"] = cand["trade_date"].astype(str)
    cand.loc[cand["trade_date"].isin(["", "nan", "None"]), "trade_date"] = trade_date

    if "ts_code" not in cand.columns:
        return cand, {"error": "Step3: candidates 缺少 ts_code，无法计算。", "trade_date": trade_date}

    snap_dir = _locate_snapshot_dir(trade_date, s=s, ctx=ctx)
    if snap_dir is None:
        out = cand.copy()
        for col in [
            "turnover_rate",
            "open_times",
            "seal_amount",
            "limit_type",
            "up_limit",
            "down_limit",
            "is_limit_up_pool",
            "first_limit_time",
            "last_limit_time",
            "limit_open_count",
        ]:
            if col not in out.columns:
                out[col] = np.nan

        out["limit_strength_raw"] = np.nan
        out["StrengthScore"] = np.nan
        out = _build_strength_quality_fields(out)

        debug["snapshot_dir"] = None
        debug["snapshot_missing"] = True
        debug["message"] = "snapshot 缺失，Step3 不伪造盘口契约字段。"
        return out, debug

    out, join_dbg = _join_snap_features(cand, snap_dir)
    debug.update(join_dbg)

    # 统一关键契约字段；保持缺失为空
    pct = _to_float_nullable(out, _first_existing_col(out, ["pct_chg", "pct_change", "change_pct", "涨跌幅"]))
    amount = _to_float_nullable(out, _first_existing_col(out, ["amount", "成交额"]))
    turnover_rate_f = _to_float_nullable(out, _first_existing_col(out, ["turnover_rate_f", "turnover_f"]))
    turnover_rate_base = _to_float_nullable(out, _first_existing_col(out, ["turnover_rate", "turn_rate", "换手率"]))
    turnover_rate = turnover_rate_f.combine_first(turnover_rate_base)

    circ_mv = _to_float_nullable(out, _first_existing_col(out, ["circ_mv", "float_mv", "流通市值"]))
    volume_ratio = _to_float_nullable(out, _first_existing_col(out, ["volume_ratio", "量比"]))
    lhb_net = _to_float_nullable(out, _first_existing_col(out, ["lhb_net_amount"]))
    hsgt_net = _to_float_nullable(out, _first_existing_col(out, ["hsgt_net_amount"]))

    seal_amount = _to_float_nullable(out, _first_existing_col(out, ["seal_amount", "fd_amount", "封单额", "封单金额"]))
    open_times = _to_float_nullable(out, _first_existing_col(out, ["open_times", "break_times", "炸板次数"]))
    limit_open_count = _to_float_nullable(out, _first_existing_col(out, ["limit_open_count", "open_times", "break_times", "炸板次数"]))
    up_limit = _to_float_nullable(out, _first_existing_col(out, ["up_limit", "涨停价"]))
    down_limit = _to_float_nullable(out, _first_existing_col(out, ["down_limit", "跌停价"]))

    limit_type_col = _first_existing_col(out, ["limit_type", "涨停类型"])
    first_limit_time_col = _first_existing_col(out, ["first_limit_time", "首次封板时间", "first_time"])
    last_limit_time_col = _first_existing_col(out, ["last_limit_time", "最后封板时间", "last_time"])
    is_limit_up_pool = _to_float_nullable(out, _first_existing_col(out, ["is_limit_up_pool", "limit_up_flag"]))

    out["turnover_rate"] = turnover_rate
    out["seal_amount"] = seal_amount
    out["open_times"] = open_times
    out["limit_open_count"] = limit_open_count
    out["up_limit"] = up_limit
    out["down_limit"] = down_limit
    out["is_limit_up_pool"] = is_limit_up_pool
    out["limit_type"] = _as_str_series(out, limit_type_col) if limit_type_col else pd.Series([pd.NA] * len(out), index=out.index)
    out["first_limit_time"] = _as_str_series(out, first_limit_time_col) if first_limit_time_col else pd.Series([pd.NA] * len(out), index=out.index)
    out["last_limit_time"] = _as_str_series(out, last_limit_time_col) if last_limit_time_col else pd.Series([pd.NA] * len(out), index=out.index)

    # 内部得分因子：允许内部计算兜底，但不污染契约字段
    pct_calc = pct.fillna(0.0)
    amount_calc = amount.fillna(0.0)
    turnover_calc = turnover_rate.fillna(0.0)
    volume_ratio_calc = volume_ratio.fillna(0.0)
    lhb_calc = lhb_net.fillna(0.0)
    hsgt_calc = hsgt_net.fillna(0.0)
    circ_mv_calc = circ_mv.fillna(0.0)
    seal_calc = seal_amount.fillna(0.0)
    open_calc = open_times.fillna(0.0)
    is_limit_calc = _safe_bool_from_flag(is_limit_up_pool)

    # 强度因子
    f_momo = _robust_rank01(pct_calc, ascending=False)
    f_amt = _robust_rank01(_log1p_safe(amount_calc), ascending=False)
    f_turn = _robust_rank01(turnover_calc, ascending=False)
    f_vr = _robust_rank01(volume_ratio_calc, ascending=False) if volume_ratio.notna().any() else pd.Series([0.0] * len(out), index=out.index)
    f_seal = _robust_rank01(_log1p_safe(seal_calc), ascending=False) if seal_amount.notna().any() else pd.Series([0.0] * len(out), index=out.index)

    # 炸板次数越多越弱
    f_open_penalty = _robust_rank01(open_calc, ascending=True) if open_times.notna().any() else pd.Series([0.0] * len(out), index=out.index)

    denom = circ_mv_calc.replace(0.0, np.nan)
    cap_raw = (lhb_calc + hsgt_calc) / denom
    cap_raw = cap_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f_cap = _robust_rank01(cap_raw, ascending=False)

    # 原始强度聚合：只表达盘口与交易强弱，不混最终标准化分
    raw01 = (
        0.18 * f_momo
        + 0.20 * f_amt
        + 0.18 * f_turn
        + 0.08 * f_vr
        + 0.16 * f_cap
        + 0.14 * f_seal
        + 0.06 * f_open_penalty
        + 0.04 * is_limit_calc
    )
    raw01 = _clip01(raw01)
    out["limit_strength_raw"] = (raw01 * 100.0).round(6)

    # 最终 StrengthScore：在原始强度基础上做轻度质量折扣
    micro_nonnull = (
        turnover_rate.notna().astype(int)
        + seal_amount.notna().astype(int)
        + open_times.notna().astype(int)
    )

    quality_adj = pd.Series([0.85] * len(out), index=out.index, dtype="float64")
    quality_adj.loc[micro_nonnull == 1] = 0.92
    quality_adj.loc[micro_nonnull >= 2] = 1.00

    strength01 = _clip01(raw01 * quality_adj)
    out["StrengthScore"] = (strength01 * 100.0).round(6)

    out["_f_momo"] = f_momo
    out["_f_amt"] = f_amt
    out["_f_turn"] = f_turn
    out["_f_vr"] = f_vr
    out["_f_cap"] = f_cap
    out["_f_seal"] = f_seal
    out["_f_open_penalty"] = f_open_penalty
    out["_strength_quality_adj"] = quality_adj

    out = _build_strength_quality_fields(out)

    # Debug
    def miss_rate(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        return float(1.0 - x.notna().mean()) if len(x) else 1.0

    def nonnull_rate(x: pd.Series) -> float:
        return float(x.notna().mean()) if len(x) else 0.0

    def nonzero_rate(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return float((x.fillna(0.0) != 0.0).mean()) if len(x) else 0.0

    debug["missing_rate"] = {
        "pct_chg": miss_rate(pct),
        "amount": miss_rate(amount),
        "turnover_rate": miss_rate(turnover_rate),
        "seal_amount": miss_rate(seal_amount),
        "open_times": miss_rate(open_times),
        "limit_type": float(1.0 - out["limit_type"].notna().mean()) if len(out) else 1.0,
        "up_limit": miss_rate(up_limit),
        "is_limit_up_pool": miss_rate(is_limit_up_pool),
        "StrengthScore": miss_rate(out["StrengthScore"]),
        "limit_strength_raw": miss_rate(out["limit_strength_raw"]),
    }
    debug["nonnull_rate"] = {
        "turnover_rate": nonnull_rate(turnover_rate),
        "seal_amount": nonnull_rate(seal_amount),
        "open_times": nonnull_rate(open_times),
        "StrengthScore": nonnull_rate(out["StrengthScore"]),
    }
    debug["nonzero_rate"] = {
        "StrengthScore": nonzero_rate(out["StrengthScore"]),
        "limit_strength_raw": nonzero_rate(out["limit_strength_raw"]),
    }
    debug["quality_distribution"] = (
        out["strength_quality_flag"].value_counts(dropna=False).to_dict()
        if "strength_quality_flag" in out.columns else {}
    )

    out = out.sort_values(
        by=["StrengthScore", "limit_strength_raw", "ts_code"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return out, debug


# =========================================================
# Debug outputs
# =========================================================

def _write_debug_outputs(trade_date: str, scored: pd.DataFrame, debug: Dict[str, Any]) -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    scored.to_csv(out_dir / f"step3_strength_{trade_date}.csv", index=False, encoding="utf-8")

    lines: List[str] = []
    lines.append("# Step3 Debug Report")
    lines.append("")
    lines.append(f"- trade_date: `{trade_date}`")
    lines.append(f"- rows: {len(scored)}")
    lines.append(f"- snapshot_dir: `{debug.get('snapshot_dir')}`")
    lines.append(f"- snapshot_missing: `{debug.get('snapshot_missing', False)}`")
    lines.append("")

    lines.append("## Files rows")
    for k, v in (debug.get("files", {}) or {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Missing rate")
    for k, v in (debug.get("missing_rate", {}) or {}).items():
        lines.append(f"- {k}: {float(v):.4f}")

    lines.append("")
    lines.append("## Nonnull rate")
    for k, v in (debug.get("nonnull_rate", {}) or {}).items():
        lines.append(f"- {k}: {float(v):.4f}")

    lines.append("")
    lines.append("## Nonzero rate")
    for k, v in (debug.get("nonzero_rate", {}) or {}).items():
        lines.append(f"- {k}: {float(v):.4f}")

    lines.append("")
    lines.append("## Strength quality distribution")
    for k, v in (debug.get("quality_distribution", {}) or {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(debug, ensure_ascii=False, indent=2))
    lines.append("```")

    (out_dir / "debug_step3_report.md").write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Public runner
# =========================================================

def run_step3(
    candidates_df: Any,
    s=None,
    ctx: Optional[Dict[str, Any]] = None,
    top_k: int = 50,
) -> pd.DataFrame:
    trade_date = _resolve_trade_date(s=s)
    scored, debug = calc_strength_score(candidates_df, trade_date=trade_date, s=s, ctx=ctx)

    try:
        _write_debug_outputs(trade_date, scored, debug)
    except Exception:
        pass

    top_k = max(1, int(top_k or 50))
    if len(scored) > top_k:
        scored = scored.head(top_k).copy()

    return scored


def run(
    df: Any,
    s=None,
    ctx: Optional[Dict[str, Any]] = None,
    top_k: int = 50,
) -> pd.DataFrame:
    return run_step3(df, s=s, ctx=ctx, top_k=top_k)


if __name__ == "__main__":
    print("Step3 StrengthScore V3 ready.")
