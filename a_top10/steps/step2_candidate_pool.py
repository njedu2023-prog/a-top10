#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step2：Candidate Pool（终版，工程对齐）
- 主程序依赖：from a_top10.steps.step2_candidate_pool import step2_build_candidates
- 关键修复：
  1) ✅ 统一 ts_code 列（兼容 ts_code/code/symbol/证券代码 等）
  2) ✅ 强制补齐 industry（来自 stock_basic.csv）
  3) ✅ 输出旁路 debug：outputs/debug_step2_candidate_YYYYMMDD.json
  4) 缺字段/缺文件自动降级，不崩溃
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


# =========================
# Utils
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_json_dump(obj: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_str_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if df is None or df.empty or not col:
        return pd.Series([""] * (0 if df is None else len(df)), dtype="string")
    return df[col].astype("string").fillna("").map(lambda x: str(x).strip())


def _normalize_ts_code_value(x: str) -> str:
    """
    兼容：
    - 000001.SZ
    - SZ000001 / SH600000
    - 000001SZ / 600000SH
    - 000001（无法判断交易所则原样返回）
    """
    s = (x or "").strip().upper()
    if not s:
        return s

    if "." in s:
        parts = s.split(".", 1)
        code = parts[0].strip()
        exch = parts[1].strip().replace("SSE", "SH").replace("SZSE", "SZ")
        if len(code) == 6 and exch in ("SH", "SZ", "BJ"):
            return f"{code}.{exch}"
        return s

    if s.startswith(("SZ", "SH", "BJ")) and len(s) >= 8:
        exch = s[:2]
        code = s[2:8]
        if code.isdigit():
            return f"{code}.{exch}"

    if len(s) >= 8 and s[:6].isdigit() and s[6:8] in ("SZ", "SH", "BJ"):
        return f"{s[:6]}.{s[6:8]}"

    return s


def _normalize_id_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    把各种代码列统一为 ts_code，并规范化值。
    """
    dbg: Dict[str, Any] = {"found_code_col": "", "normalized_ratio": 0.0}
    if df is None or df.empty:
        return df, dbg

    code_col = _first_existing_col(
        df,
        ["ts_code", "TS_CODE", "code", "CODE", "symbol", "SYMBOL", "证券代码", "代码", "股票代码"],
    )
    dbg["found_code_col"] = code_col or ""
    if not code_col:
        return df, dbg

    out = df.copy()
    out["ts_code"] = _to_str_series(out, code_col).map(_normalize_ts_code_value)

    dbg["normalized_ratio"] = float((out["ts_code"].astype("string").fillna("") != "").mean())
    return out, dbg


def _resolve_outputs_dir(s: Settings) -> Path:
    for key in ["outputs_dir", "output_dir", "outputs", "out_dir"]:
        if hasattr(s, key):
            v = getattr(s, key)
            try:
                return Path(v)
            except Exception:
                pass
    return Path("outputs")


def _pick_trade_date(s: Settings, ctx: Dict[str, Any]) -> str:
    td = str(ctx.get("trade_date", "") or "").strip()
    if td:
        return td
    if hasattr(s, "trade_date") and getattr(s, "trade_date"):
        return str(getattr(s, "trade_date")).strip()
    return "unknown"


def _ctx_get_df(ctx: Dict[str, Any], keys: List[str]) -> Optional[pd.DataFrame]:
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


# =========================
# Step2 main
# =========================

def step2_build_candidates(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    输出：
      ctx["candidates"] = candidates_df
      ctx["step2"] = candidates_df   # 兼容旧链路
    """
    trade_date = _pick_trade_date(s, ctx)
    out_dir = _resolve_outputs_dir(s)
    _ensure_dir(out_dir)

    debug: Dict[str, Any] = {
        "trade_date": trade_date,
        "base_source": "",
        "base_rows": 0,
        "code_norm": {},
        "stock_basic_rows": 0,
        "industry_merge": {
            "ok": False,
            "reason": "",
            "industry_nonblank_ratio_before": 0.0,
            "industry_nonblank_ratio_after": 0.0,
        },
        "filters": {"drop_st": 0, "drop_delist": 0},
        "final_rows": 0,
        "final_cols": [],
        "debug_file": str(out_dir / f"debug_step2_candidate_{trade_date}.json"),
        "out_csv": str(out_dir / f"step2_candidates_{trade_date}.csv"),
    }

    # 1) 选择 base：优先 limit_list_d / stk_limit（涨停列表）
    base_df = _ctx_get_df(ctx, ["limit_list_d", "stk_limit", "limit_up", "limit_up_list"])
    if base_df is not None and not base_df.empty:
        debug["base_source"] = "limit_list_d/stk_limit"
    else:
        base_df = _ctx_get_df(ctx, ["top_list", "daily", "daily_basic"])
        debug["base_source"] = "top_list/daily/daily_basic" if base_df is not None else ""

    if base_df is None or base_df.empty:
        empty = pd.DataFrame()
        ctx["candidates"] = empty
        ctx["step2"] = empty
        _safe_json_dump(debug, Path(debug["debug_file"]))
        return ctx

    base_df = base_df.copy()
    debug["base_rows"] = int(len(base_df))

    # 2) 统一 ts_code
    base_df, code_dbg = _normalize_id_columns(base_df)
    debug["code_norm"] = code_dbg

    # 3) 若没有 ts_code，直接输出（让问题显性暴露）
    if "ts_code" not in base_df.columns:
        candidates_df = base_df
        debug["final_rows"] = int(len(candidates_df))
        debug["final_cols"] = list(candidates_df.columns)
        candidates_df.to_csv(Path(debug["out_csv"]), index=False, encoding="utf-8-sig")
        _safe_json_dump(debug, Path(debug["debug_file"]))
        ctx["candidates"] = candidates_df
        ctx["step2"] = candidates_df
        return ctx

    # 4) industry 补齐（关键）
    before_ratio = 0.0
    if "industry" in base_df.columns:
        before_ratio = float((base_df["industry"].astype("string").fillna("") != "").mean())
    debug["industry_merge"]["industry_nonblank_ratio_before"] = before_ratio

    sb = _ctx_get_df(ctx, ["stock_basic", "stock_basic_df"])
    if sb is None or sb.empty:
        debug["industry_merge"]["ok"] = False
        debug["industry_merge"]["reason"] = "stock_basic missing in ctx"
        candidates_df = base_df
    else:
        sb = sb.copy()
        debug["stock_basic_rows"] = int(len(sb))

        sb, sb_code_dbg = _normalize_id_columns(sb)
        sb_ind_col = _first_existing_col(sb, ["industry", "行业", "industry_name"])
        if not sb_ind_col:
            debug["industry_merge"]["ok"] = False
            debug["industry_merge"]["reason"] = "industry col missing in stock_basic"
            candidates_df = base_df
        else:
            sb2 = sb[["ts_code", sb_ind_col]].rename(columns={sb_ind_col: "industry"}).copy()
            sb2["industry"] = sb2["industry"].astype("string").fillna("").map(lambda x: str(x).strip())

            candidates_df = base_df.merge(sb2, on="ts_code", how="left", suffixes=("", "_sb"))

            if "industry_sb" in candidates_df.columns:
                if "industry" in candidates_df.columns:
                    base_ind = candidates_df["industry"].astype("string").fillna("")
                    sb_ind = candidates_df["industry_sb"].astype("string").fillna("")
                    candidates_df["industry"] = base_ind.where(base_ind != "", sb_ind)
                else:
                    candidates_df["industry"] = candidates_df["industry_sb"]
                candidates_df.drop(columns=["industry_sb"], inplace=True, errors="ignore")

            debug["industry_merge"]["ok"] = True
            debug["industry_merge"]["reason"] = ""
            debug["industry_merge"]["stock_basic_code_norm"] = sb_code_dbg

    after_ratio = 0.0
    if candidates_df is not None and not candidates_df.empty and "industry" in candidates_df.columns:
        after_ratio = float((candidates_df["industry"].astype("string").fillna("") != "").mean())
    debug["industry_merge"]["industry_nonblank_ratio_after"] = after_ratio

    # 5) 基础过滤：ST / 退（可选，缺字段就跳过）
    name_col = _first_existing_col(candidates_df, ["name", "名称", "股票简称", "ts_name"])
    if name_col:
        name_s = _to_str_series(candidates_df, name_col)

        mask_st = name_s.str.contains("ST", case=False, regex=False)
        debug["filters"]["drop_st"] = int(mask_st.sum())
        candidates_df = candidates_df.loc[~mask_st].copy()

        name_s2 = _to_str_series(candidates_df, name_col)
        mask_delist = name_s2.str.contains("退", regex=False) | name_s2.str.contains("退市", regex=False)
        debug["filters"]["drop_delist"] = int(mask_delist.sum())
        candidates_df = candidates_df.loc[~mask_delist].copy()

    # 6) 去重与列前置
    candidates_df["ts_code"] = candidates_df["ts_code"].astype("string").fillna("").map(_normalize_ts_code_value)
    candidates_df = candidates_df.loc[candidates_df["ts_code"] != ""].copy()
    candidates_df = candidates_df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)

    front = [c for c in ["ts_code", "name", "industry"] if c in candidates_df.columns]
    rest = [c for c in candidates_df.columns if c not in front]
    candidates_df = candidates_df[front + rest]

    # 7) 落盘 + debug
    debug["final_rows"] = int(len(candidates_df))
    debug["final_cols"] = list(candidates_df.columns)

    candidates_df.to_csv(Path(debug["out_csv"]), index=False, encoding="utf-8-sig")
    _safe_json_dump(debug, Path(debug["debug_file"]))

    # 8) 写回 ctx
    ctx["candidates"] = candidates_df
    ctx["step2"] = candidates_df

    return ctx
