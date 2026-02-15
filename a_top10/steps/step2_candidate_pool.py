#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step2：候选池构建 / 过滤（终版）
目标（工程硬指标）：
1) 输出候选池 candidates_df（供 Step3~Step7 使用）
2) ✅ 强制补齐 industry（来自 stock_basic.csv）
3) ✅ 统一 ts_code 列名（不管上游叫 ts_code / code / symbol / 证券代码）
4) 兼容字段缺失：缺什么就降级，不报错
5) 产出稳定旁路调试：outputs/debug_step2_candidate_YYYYMMDD.json

重要：Step4 题材加成依赖：
- 候选池必须有 ts_code
- 候选池必须有 industry
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from a_top10.config import Settings


# -------------------------
# utils
# -------------------------

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
        return pd.Series([], dtype="string")
    return df[col].astype("string").fillna("").map(lambda x: str(x).strip())


def _normalize_ts_code_value(x: str) -> str:
    """
    兼容：
    000001.SZ / 000001sz / SZ000001 / 000001
    """
    s = (x or "").strip().upper()
    if not s:
        return s

    # 已是 000001.SZ / 000001.SH
    if "." in s and len(s.split(".")[0]) in (6,):
        code, exch = s.split(".", 1)
        exch = exch.replace("SSE", "SH").replace("SZSE", "SZ")
        if exch in ("SH", "SZ", "BJ"):
            return f"{code}.{exch}"

    # SZ000001 / SH600000
    if s.startswith(("SZ", "SH", "BJ")) and len(s) >= 8:
        exch = s[:2]
        code = s[2:8]
        if code.isdigit():
            return f"{code}.{exch}"

    # 000001SZ / 000001SH
    if len(s) >= 8 and s[:6].isdigit() and s[6:8] in ("SZ", "SH", "BJ"):
        return f"{s[:6]}.{s[6:8]}"

    # 仅 6 位数字：无法确定交易所 -> 原样返回（后续 merge 会尽力）
    return s


def _normalize_id_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    把各种代码列统一为 ts_code，并把值规范化。
    """
    debug: Dict[str, Any] = {"found_code_col": "", "normalized_ratio": 0.0}

    if df is None or df.empty:
        return df, debug

    code_col = _first_existing_col(
        df,
        ["ts_code", "TS_CODE", "code", "CODE", "symbol", "SYMBOL", "证券代码", "代码", "股票代码"],
    )
    debug["found_code_col"] = code_col or ""

    if not code_col:
        # 保留原 df，后续步骤会看到 ts_code 缺失
        return df, debug

    s = _to_str_series(df, code_col).map(_normalize_ts_code_value)
    out = df.copy()
    out["ts_code"] = s

    nonblank = (out["ts_code"].astype("string").fillna("") != "").mean()
    debug["normalized_ratio"] = float(nonblank)

    return out, debug


def _resolve_outputs_dir(s: Settings) -> Path:
    # 兼容 Settings 不同版本字段名
    for key in ["outputs_dir", "output_dir", "outputs", "out_dir"]:
        if hasattr(s, key):
            v = getattr(s, key)
            try:
                return Path(v)
            except Exception:
                pass
    return Path("outputs")


def _ctx_get_df(ctx: Dict[str, Any], keys: List[str]) -> Optional[pd.DataFrame]:
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


def _pick_trade_date(s: Settings, ctx: Dict[str, Any]) -> str:
    # 兼容：ctx / Settings / ENV 已在主程序处理；这里兜底
    td = str(ctx.get("trade_date", "") or "").strip()
    if td:
        return td
    if hasattr(s, "trade_date") and getattr(s, "trade_date"):
        return str(getattr(s, "trade_date")).strip()
    return "unknown"


# -------------------------
# core
# -------------------------

def step2_filter(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入（ctx 里来自 Step0/Step1 的数据，尽量宽松）：
      - limit_list_d / stk_limit / top_list / daily / daily_basic 等
      - stock_basic（用于补 industry）

    输出：
      ctx["candidates"] = candidates_df
      ctx["step2"] = candidates_df   # 双写兼容旧主程序
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
        "filters": {
            "drop_st": 0,
            "drop_delist": 0,
        },
        "final_rows": 0,
        "final_cols": [],
        "debug_file": str(out_dir / f"debug_step2_candidate_{trade_date}.json"),
        "out_csv": str(out_dir / f"step2_candidates_{trade_date}.csv"),
    }

    # 1) 选择候选池 base：优先使用“涨停列表”
    base_df = _ctx_get_df(ctx, ["limit_list_d", "stk_limit", "limit_up", "limit_up_list"])
    if base_df is not None and not base_df.empty:
        debug["base_source"] = "limit_list_d/stk_limit"
    else:
        # 降级：用 top_list 或 daily（尽量给 Step3/4 一个候选集合）
        base_df = _ctx_get_df(ctx, ["top_list", "daily", "daily_basic"])
        debug["base_source"] = "top_list/daily/daily_basic" if base_df is not None else ""

    if base_df is None or base_df.empty:
        # 彻底没数据：返回空候选池，但写 debug
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

    # 3) 如果没有 ts_code，直接输出（后续会显性暴露问题）
    if "ts_code" not in base_df.columns:
        candidates_df = base_df
        debug["final_rows"] = int(len(candidates_df))
        debug["final_cols"] = list(candidates_df.columns)
        candidates_df.to_csv(Path(debug["out_csv"]), index=False, encoding="utf-8-sig")
        _safe_json_dump(debug, Path(debug["debug_file"]))
        ctx["candidates"] = candidates_df
        ctx["step2"] = candidates_df
        return ctx

    # 4) 补齐 industry（关键修复点）
    industry_before = 0.0
    if "industry" in base_df.columns:
        industry_before = float((base_df["industry"].astype("string").fillna("") != "").mean())
    debug["industry_merge"]["industry_nonblank_ratio_before"] = industry_before

    sb = _ctx_get_df(ctx, ["stock_basic", "stock_basic_df"])
    if sb is None or sb.empty:
        debug["industry_merge"]["ok"] = False
        debug["industry_merge"]["reason"] = "stock_basic missing in ctx"
        candidates_df = base_df
    else:
        sb = sb.copy()
        debug["stock_basic_rows"] = int(len(sb))

        # stock_basic 也统一 ts_code
        sb, sb_code_dbg = _normalize_id_columns(sb)
        # 取 industry 字段
        sb_industry_col = _first_existing_col(sb, ["industry", "行业", "industry_name"])
        if not sb_industry_col:
            debug["industry_merge"]["ok"] = False
            debug["industry_merge"]["reason"] = "industry col missing in stock_basic"
            candidates_df = base_df
        else:
            # 清洗：只保留 ts_code + industry（避免列冲突）
            sb2 = sb[["ts_code", sb_industry_col]].rename(columns={sb_industry_col: "industry"}).copy()
            sb2["industry"] = sb2["industry"].astype("string").fillna("").map(lambda x: str(x).strip())

            # 左连接补齐
            candidates_df = base_df.merge(sb2, on="ts_code", how="left", suffixes=("", "_sb"))
            # 若 base 自带 industry 列，则优先 base；否则用 sb
            if "industry_sb" in candidates_df.columns:
                if "industry" in candidates_df.columns:
                    # base 的 industry 为空时才用 sb
                    base_ind = candidates_df["industry"].astype("string").fillna("")
                    sb_ind = candidates_df["industry_sb"].astype("string").fillna("")
                    candidates_df["industry"] = base_ind.where(base_ind != "", sb_ind)
                else:
                    candidates_df["industry"] = candidates_df["industry_sb"]
                candidates_df.drop(columns=["industry_sb"], inplace=True, errors="ignore")

            debug["industry_merge"]["ok"] = True
            debug["industry_merge"]["reason"] = ""
            debug["industry_merge"]["stock_basic_code_norm"] = sb_code_dbg

    industry_after = 0.0
    if candidates_df is not None and not candidates_df.empty and "industry" in candidates_df.columns:
        industry_after = float((candidates_df["industry"].astype("string").fillna("") != "").mean())
    debug["industry_merge"]["industry_nonblank_ratio_after"] = industry_after

    # 5) 基础过滤：剔除 ST / 退市风险（尽量不误杀，缺字段就跳过）
    name_col = _first_existing_col(candidates_df, ["name", "名称", "股票简称", "ts_name"])
    if name_col:
        name_s = _to_str_series(candidates_df, name_col)

        # ST
        mask_st = name_s.str.contains("ST", case=False, regex=False)
        st_cnt = int(mask_st.sum())
        debug["filters"]["drop_st"] = st_cnt
        candidates_df = candidates_df.loc[~mask_st].copy()

        # 退市风险：*退 / 退市 / DR 等（保守规则，避免过拟合）
        name_s2 = _to_str_series(candidates_df, name_col)
        mask_delist = name_s2.str.contains("退", regex=False) | name_s2.str.contains("退市", regex=False)
        del_cnt = int(mask_delist.sum())
        debug["filters"]["drop_delist"] = del_cnt
        candidates_df = candidates_df.loc[~mask_delist].copy()

    # 6) 去重与稳定列顺序
    candidates_df["ts_code"] = candidates_df["ts_code"].astype("string").fillna("").map(_normalize_ts_code_value)
    candidates_df = candidates_df.loc[candidates_df["ts_code"] != ""].copy()
    candidates_df = candidates_df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)

    # 尽量把核心列前置（不影响其它列存在）
    front = [c for c in ["ts_code", "name", "industry", "close", "pct_chg", "pct_change", "turnover_rate", "amount"] if c in candidates_df.columns]
    rest = [c for c in candidates_df.columns if c not in front]
    candidates_df = candidates_df[front + rest]

    # 7) 输出落盘 + debug
    debug["final_rows"] = int(len(candidates_df))
    debug["final_cols"] = list(candidates_df.columns)

    candidates_df.to_csv(Path(debug["out_csv"]), index=False, encoding="utf-8-sig")
    _safe_json_dump(debug, Path(debug["debug_file"]))

    # 8) 写回 ctx（双 key 兼容）
    ctx["candidates"] = candidates_df
    ctx["step2"] = candidates_df

    return ctx
