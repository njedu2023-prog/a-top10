#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step3: Strength Score — Engine V2 (基于可用数据的真实量化版)

目标：
1) 读取 Step2 候选池（或上游传入 DataFrame/dict）
2) 自动定位数据仓库快照目录，读取可用 CSV（daily / daily_basic / top_list / moneyflow_hsgt / limit_list_d / limit_break_d）
3) 用 ts_code(+trade_date) join 补齐字段
4) 计算 StrengthScore（0~100），并输出调试分项，保证 Step6 能识别 StrengthScore
5) ✅ 关键：尽可能补齐并透传 Step5/feature_history 需要的核心字段：
   - turnover_rate
   - seal_amount
   - open_times

输出：
- 返回：带 StrengthScore 的 DataFrame（默认 TopK=50）
- 旁路落盘（不影响主流程）：
  outputs/debug_step3_report.md
  outputs/step3_strength_YYYYMMDD.csv
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Basic utils
# =========================================================

def _coerce_df(obj: Any) -> pd.DataFrame:
    """统一把输入转换成 DataFrame：兼容 DataFrame / dict(内含 DataFrame) / None"""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, Mapping):
        for k in ("candidates", "candidate", "df", "data", "result", "pool", "step2", "step2_candidates"):
            v = obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame()
    return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if (col is None) or (col not in df.columns):
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


def _as_str_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: pd.NA if isinstance(x, str) and x == "" else x)
    return s


def _clip01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.clip(0.0, 1.0)


def _robust_rank01(s: pd.Series, ascending: bool = False) -> pd.Series:
    """
    稳健归一（0~1）：用 rank(pct=True) 抗尺度/长尾。
    ascending=False 表示值越大越强
    """
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
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
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d")


# =========================================================
# Snapshot locating (不写死，自动探测)
# =========================================================

def _candidate_snapshot_dirs(trade_date: str) -> Sequence[Path]:
    y = trade_date[:4]
    return [
        Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / trade_date,
        Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / trade_date,
        Path("data_repo") / "snapshots" / trade_date,
        Path("snapshots") / trade_date,
    ]


def _locate_snapshot_dir(trade_date: str, s=None, ctx: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    if ctx and isinstance(ctx, dict):
        p = ctx.get("snapshot_dir", None)
        if p:
            pp = Path(str(p))
            if pp.exists():
                return pp

    if s is not None and hasattr(s, "snapshot_dir"):
        try:
            pp = s.snapshot_dir(trade_date)
            pp = Path(str(pp))
            if pp.exists():
                return pp
        except Exception:
            pass

    for p in _candidate_snapshot_dirs(trade_date):
        if p.exists():
            return p

    return None


def _read_csv(p: Path) -> pd.DataFrame:
    if not p or (not p.exists()):
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, encoding="utf-8", errors="ignore")
        except Exception:
            try:
                return pd.read_csv(p, encoding="gbk", errors="ignore")
            except Exception:
                return pd.DataFrame()


# =========================================================
# Candidate pool loading
# =========================================================

def _load_candidates_fallback(trade_date: str) -> pd.DataFrame:
    p = Path("outputs") / f"step2_candidates_{trade_date}.csv"
    if p.exists():
        return _read_csv(p)
    return pd.DataFrame()


def _ensure_identity(out: pd.DataFrame) -> pd.DataFrame:
    if out is None or out.empty:
        return out

    ts_col = _first_existing_col(out, ["ts_code", "TS_CODE", "code", "symbol", "ticker", "证券代码", "代码"]) or "ts_code"
    if ts_col != "ts_code":
        out["ts_code"] = _as_str_series(out, ts_col)
    else:
        out["ts_code"] = _as_str_series(out, "ts_code")

    name_col = _first_existing_col(out, ["name", "NAME", "stock_name", "名称", "股票", "证券名称"])
    if name_col:
        out["name"] = _as_str_series(out, name_col)
    elif "name" not in out.columns:
        out["name"] = ""

    ind_col = _first_existing_col(out, ["industry", "行业", "板块", "industry_name", "所属行业"])
    if ind_col:
        out["industry"] = _as_str_series(out, ind_col)
    elif "industry" not in out.columns:
        out["industry"] = ""

    td_col = _first_existing_col(out, ["trade_date", "TRADE_DATE", "日期"])
    if td_col:
        out["trade_date"] = out[td_col].astype(str)
    elif "trade_date" not in out.columns:
        out["trade_date"] = ""

    return out


# =========================================================
# Feature join from snapshots
# =========================================================

def _normalize_ts_code(df: pd.DataFrame, col: str = "ts_code") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.strip()
    return df


def _pick_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return _first_existing_col(df, names)


def _extract_single_value_by_day(
    df: pd.DataFrame,
    prefix: str,
    value_candidates: Sequence[str],
) -> pd.DataFrame:
    """
    从 df 里提取一个值列（例如 seal_amount/open_times），按 ts_code+trade_date 聚合：
    - 数值列：sum（同一天多行时）
    """
    if df is None or df.empty or ("ts_code" not in df.columns) or ("trade_date" not in df.columns):
        return pd.DataFrame()

    val_col = _pick_col(df, value_candidates)
    if not val_col:
        return pd.DataFrame()

    x = df[["ts_code", "trade_date", val_col]].copy()
    out_col = f"{prefix}"
    x = x.rename(columns={val_col: out_col})
    x[out_col] = pd.to_numeric(x[out_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.groupby(["ts_code", "trade_date"], as_index=False)[out_col].sum()
    return x


def _join_snap_features(cand: pd.DataFrame, snap_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    从快照目录读取多个 csv，并按 ts_code+trade_date 合并到候选池。
    """
    debug: Dict[str, Any] = {"snapshot_dir": str(snap_dir), "files": {}}

    daily = _read_csv(snap_dir / "daily.csv")
    daily_basic = _read_csv(snap_dir / "daily_basic.csv")
    top_list = _read_csv(snap_dir / "top_list.csv")
    hsgt = _read_csv(snap_dir / "moneyflow_hsgt.csv")
    limit_list = _read_csv(snap_dir / "limit_list_d.csv")
    limit_break = _read_csv(snap_dir / "limit_break_d.csv")

    debug["files"]["daily.csv"] = int(len(daily))
    debug["files"]["daily_basic.csv"] = int(len(daily_basic))
    debug["files"]["top_list.csv"] = int(len(top_list))
    debug["files"]["moneyflow_hsgt.csv"] = int(len(hsgt))
    debug["files"]["limit_list_d.csv"] = int(len(limit_list))
    debug["files"]["limit_break_d.csv"] = int(len(limit_break))

    cand = cand.copy()
    cand = _normalize_ts_code(cand, "ts_code")

    for df in (daily, daily_basic, top_list, hsgt, limit_list, limit_break):
        if not df.empty and "ts_code" in df.columns:
            _normalize_ts_code(df, "ts_code")

    def norm_td(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        td_col = _pick_col(df, ["trade_date", "TRADE_DATE", "日期"])
        if td_col and td_col != "trade_date":
            df = df.rename(columns={td_col: "trade_date"})
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(str)
        return df

    daily = norm_td(daily)
    daily_basic = norm_td(daily_basic)
    top_list = norm_td(top_list)
    hsgt = norm_td(hsgt)
    limit_list = norm_td(limit_list)
    limit_break = norm_td(limit_break)

    # 合并 daily：pct_chg/amount/vol
    if not daily.empty and ("ts_code" in daily.columns) and ("trade_date" in daily.columns):
        pct_col = _pick_col(daily, ["pct_chg", "pct_change", "change_pct", "涨跌幅"])
        amt_col = _pick_col(daily, ["amount", "成交额", "turnover_amount", "amt"])
        vol_col = _pick_col(daily, ["vol", "成交量", "volume"])
        keep = ["ts_code", "trade_date"] + [c for c in [pct_col, amt_col, vol_col] if c]
        d = daily[keep].copy()
        if pct_col and pct_col != "pct_chg":
            d = d.rename(columns={pct_col: "pct_chg"})
        if amt_col and amt_col != "amount":
            d = d.rename(columns={amt_col: "amount"})
        if vol_col and vol_col != "vol":
            d = d.rename(columns={vol_col: "vol"})
        cand = cand.merge(d, on=["ts_code", "trade_date"], how="left")

    # 合并 daily_basic：turnover_rate/circ_mv/volume_ratio...
    if not daily_basic.empty and ("ts_code" in daily_basic.columns) and ("trade_date" in daily_basic.columns):
        trf_col = _pick_col(daily_basic, ["turnover_rate_f", "turnover_f"])
        tr_col = _pick_col(daily_basic, ["turnover_rate", "turn_rate", "换手率"])
        cmv_col = _pick_col(daily_basic, ["circ_mv", "float_mv", "流通市值"])
        tmv_col = _pick_col(daily_basic, ["total_mv", "总市值"])
        vr_col = _pick_col(daily_basic, ["volume_ratio", "量比"])
        keep = ["ts_code", "trade_date"] + [c for c in [trf_col, tr_col, cmv_col, tmv_col, vr_col] if c]
        db = daily_basic[keep].copy()
        if trf_col and trf_col != "turnover_rate_f":
            db = db.rename(columns={trf_col: "turnover_rate_f"})
        if tr_col and tr_col != "turnover_rate":
            db = db.rename(columns={tr_col: "turnover_rate"})
        if cmv_col and cmv_col != "circ_mv":
            db = db.rename(columns={cmv_col: "circ_mv"})
        if tmv_col and tmv_col != "total_mv":
            db = db.rename(columns={tmv_col: "total_mv"})
        if vr_col and vr_col != "volume_ratio":
            db = db.rename(columns={vr_col: "volume_ratio"})
        cand = cand.merge(db, on=["ts_code", "trade_date"], how="left")

    # 龙虎榜资金净额（top_list）/ 北向（moneyflow_hsgt）：自动找 net_* 字段并聚合
    def extract_net_amount(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df is None or df.empty or ("ts_code" not in df.columns) or ("trade_date" not in df.columns):
            return pd.DataFrame()
        cand_cols = [c for c in df.columns if "net" in str(c).lower()]
        net_col = _pick_col(df, ["net_amount", "net_amt", "net_buy", "net"]) or (cand_cols[0] if cand_cols else None)
        if not net_col:
            return pd.DataFrame()
        x = df[["ts_code", "trade_date", net_col]].copy()
        x = x.rename(columns={net_col: f"{prefix}_net_amount"})
        x[f"{prefix}_net_amount"] = pd.to_numeric(x[f"{prefix}_net_amount"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x = x.groupby(["ts_code", "trade_date"], as_index=False)[f"{prefix}_net_amount"].sum()
        return x

    tl = extract_net_amount(top_list, "lhb")
    if not tl.empty:
        cand = cand.merge(tl, on=["ts_code", "trade_date"], how="left")

    hs = extract_net_amount(hsgt, "hsgt")
    if not hs.empty:
        cand = cand.merge(hs, on=["ts_code", "trade_date"], how="left")

    # 涨停/炸板标记
    if not limit_list.empty and ("ts_code" in limit_list.columns) and ("trade_date" in limit_list.columns):
        ll = limit_list[["ts_code", "trade_date"]].drop_duplicates().copy()
        ll["limit_up_flag"] = 1
        cand = cand.merge(ll, on=["ts_code", "trade_date"], how="left")

    if not limit_break.empty and ("ts_code" in limit_break.columns) and ("trade_date" in limit_break.columns):
        lb = limit_break[["ts_code", "trade_date"]].drop_duplicates().copy()
        lb["limit_break_flag"] = 1
        cand = cand.merge(lb, on=["ts_code", "trade_date"], how="left")

    # ✅ 关键：尝试抽取 seal_amount / open_times（多候选字段名自动适配）
    # 不保证每个仓库一定有，但只要存在，就能向下游提供“真实非零信息”
    seal = _extract_single_value_by_day(
        limit_list,
        prefix="seal_amount",
        value_candidates=[
            "seal_amount", "fd_amount", "fund_amount", "封单额", "封单金额", "封单资金", "order_amount", "amount",
        ],
    )
    if not seal.empty:
        cand = cand.merge(seal, on=["ts_code", "trade_date"], how="left")

    opens = _extract_single_value_by_day(
        limit_break,
        prefix="open_times",
        value_candidates=[
            "open_times", "open_num", "break_times", "break_count", "炸板次数", "open_cnt", "times",
        ],
    )
    if not opens.empty:
        cand = cand.merge(opens, on=["ts_code", "trade_date"], how="left")

    return cand, debug


# =========================================================
# Core scoring (真实量化：基于可用数据)
# =========================================================

def calc_strength_score(df: Any, trade_date: str, s=None, ctx: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    返回：(scored_df, debug_info)
    scored_df 至少包含：ts_code / name / industry / trade_date / StrengthScore
    且尽可能包含：turnover_rate / seal_amount / open_times（用于 Step5/feature_history）
    """
    debug: Dict[str, Any] = {"trade_date": trade_date}

    cand = _coerce_df(df)
    if cand.empty:
        cand = _load_candidates_fallback(trade_date)

    cand = _ensure_identity(cand)

    # trade_date 兜底填充
    if "trade_date" not in cand.columns:
        cand["trade_date"] = trade_date
    cand["trade_date"] = cand["trade_date"].astype(str)
    cand.loc[cand["trade_date"].isin(["", "nan", "None"]), "trade_date"] = trade_date

    if "ts_code" not in cand.columns:
        return cand, {"error": "Step3: candidates 缺少 ts_code，无法计算。"}

    snap_dir = _locate_snapshot_dir(trade_date, s=s, ctx=ctx)
    if snap_dir is None:
        out = cand.copy()
        out["StrengthScore"] = 0.0
        # ✅ 确保标准列存在，避免下游全缺列
        out["turnover_rate"] = 0.0
        out["seal_amount"] = 0.0
        out["open_times"] = 0.0
        debug["snapshot_dir"] = None
        debug["snapshot_missing"] = True
        return out, debug

    out, join_dbg = _join_snap_features(cand, snap_dir)
    debug.update(join_dbg)

    # ---------- 统一数值列 ----------
    pct = _to_float_series(out, _first_existing_col(out, ["pct_chg", "pct_change", "change_pct", "涨跌幅"]), 0.0)
    amount = _to_float_series(out, _first_existing_col(out, ["amount", "成交额"]), 0.0)

    trf = _to_float_series(out, _first_existing_col(out, ["turnover_rate_f"]), np.nan)
    tr = _to_float_series(out, _first_existing_col(out, ["turnover_rate", "turn_rate", "换手率"]), np.nan)
    turnover = trf.where(trf.notna() & (trf > 0), tr).fillna(0.0)

    circ_mv = _to_float_series(out, _first_existing_col(out, ["circ_mv", "float_mv", "流通市值"]), np.nan).fillna(0.0)
    volume_ratio = _to_float_series(out, _first_existing_col(out, ["volume_ratio", "量比"]), np.nan).fillna(0.0)

    lhb_net = _to_float_series(out, _first_existing_col(out, ["lhb_net_amount"]), 0.0)
    hsgt_net = _to_float_series(out, _first_existing_col(out, ["hsgt_net_amount"]), 0.0)

    limit_up_flag = _to_float_series(out, _first_existing_col(out, ["limit_up_flag"]), 0.0)
    limit_break_flag = _to_float_series(out, _first_existing_col(out, ["limit_break_flag"]), 0.0)

    # ✅ 关键：把 seal_amount/open_times 标准化并确保存在
    seal_amount = _to_float_series(out, _first_existing_col(out, ["seal_amount", "fd_amount", "封单额", "封单金额"]), 0.0)
    open_times = _to_float_series(out, _first_existing_col(out, ["open_times", "break_times", "炸板次数"]), 0.0)

    # 把标准列写回（保证下游拿得到）
    out["turnover_rate"] = turnover
    out["seal_amount"] = seal_amount
    out["open_times"] = open_times

    # ---------- 构造稳健因子（0~1） ----------
    f_momo = _robust_rank01(pct, ascending=False)
    f_amt = _robust_rank01(_log1p_safe(amount), ascending=False)
    f_turn = _robust_rank01(turnover, ascending=False)

    # 量比
    f_vr = _robust_rank01(volume_ratio, ascending=False) if volume_ratio.notna().any() else pd.Series([0.0] * len(out), index=out.index)

    # 资金强度：龙虎榜+北向 / 流通市值
    denom = circ_mv.replace(0.0, np.nan)
    cap_raw = (lhb_net.fillna(0.0) + hsgt_net.fillna(0.0)) / denom
    cap_raw = cap_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f_cap = _robust_rank01(cap_raw, ascending=False)

    # 小市值偏好
    mv_raw = circ_mv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f_small = _robust_rank01(mv_raw, ascending=True)

    bonus = 0.05 * (limit_up_flag > 0.5).astype("float64")
    penalty = 0.07 * (limit_break_flag > 0.5).astype("float64")

    # 合成权重（可解释、可调）
    w_momo = 0.18
    w_amt = 0.26
    w_turn = 0.20
    w_vr = 0.08
    w_cap = 0.18
    w_small = 0.10

    score01 = (
        w_momo * f_momo
        + w_amt * f_amt
        + w_turn * f_turn
        + w_vr * f_vr
        + w_cap * f_cap
        + w_small * f_small
        + bonus
        - penalty
    )

    score01 = _clip01(score01)

    # 写回分项（便于调试/回测）
    out["_f_momo"] = f_momo
    out["_f_amt"] = f_amt
    out["_f_turn"] = f_turn
    out["_f_vr"] = f_vr
    out["_f_cap"] = f_cap
    out["_f_small"] = f_small
    out["_bonus"] = bonus
    out["_penalty"] = penalty

    out["StrengthScore"] = (score01 * 100.0).round(6)

    # ---------- Debug：缺失率 + 非零率 ----------
    def miss_rate(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        return float(1.0 - x.notna().mean()) if len(x) else 1.0

    def nonzero_rate(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float((x != 0.0).mean()) if len(x) else 0.0

    debug["missing_rate"] = {
        "pct_chg": miss_rate(pct),
        "amount": miss_rate(amount),
        "turnover_rate": miss_rate(turnover),
        "circ_mv": miss_rate(circ_mv),
        "volume_ratio": miss_rate(volume_ratio),
        "lhb_net_amount": miss_rate(lhb_net),
        "hsgt_net_amount": miss_rate(hsgt_net),
        "seal_amount": miss_rate(seal_amount),
        "open_times": miss_rate(open_times),
    }
    debug["nonzero_rate"] = {
        "StrengthScore": nonzero_rate(out["StrengthScore"]),
        "turnover_rate": nonzero_rate(turnover),
        "seal_amount": nonzero_rate(seal_amount),
        "open_times": nonzero_rate(open_times),
    }

    out = out.sort_values("StrengthScore", ascending=False).reset_index(drop=True)
    return out, debug


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
    files = debug.get("files", {}) or {}
    for k, v in files.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Missing rate")
    mr = debug.get("missing_rate", {}) or {}
    for k, v in mr.items():
        try:
            lines.append(f"- {k}: {float(v):.4f}")
        except Exception:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Nonzero rate (关键验收)")
    nz = debug.get("nonzero_rate", {}) or {}
    for k, v in nz.items():
        try:
            lines.append(f"- {k}: {float(v):.4f}")
        except Exception:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(debug, ensure_ascii=False, indent=2))
    lines.append("```")

    (out_dir / "debug_step3_report.md").write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Public runner (与原系统对接)
# =========================================================

def run_step3(candidates_df: Any, s=None, ctx: Optional[Dict[str, Any]] = None, top_k: int = 50) -> pd.DataFrame:
    trade_date = _resolve_trade_date(s=s)
    scored, debug = calc_strength_score(candidates_df, trade_date=trade_date, s=s, ctx=ctx)

    # 旁路落盘（不影响主线）
    try:
        _write_debug_outputs(trade_date, scored, debug)
    except Exception:
        pass

    top_k = int(top_k or 50)
    top_k = max(1, top_k)
    if len(scored) > top_k:
        scored = scored.head(top_k).copy()

    return scored


def run(df: Any, s=None, ctx: Optional[Dict[str, Any]] = None, top_k: int = 50) -> pd.DataFrame:
    """向后兼容入口"""
    return run_step3(df, s=s, ctx=ctx, top_k=top_k)


if __name__ == "__main__":
    print("Step3 StrengthScore V2 ready.")
