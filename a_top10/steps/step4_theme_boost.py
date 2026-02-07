#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4 : 题材/板块加权（ThemeBoost）

目标（0204-TOP10）：
- 输出字段：题材加成（0~1）
- 先用 hot_boards.csv（行业热度）跑通最小闭环
- 若缺行业字段，自动从 stock_basic 补齐
- 全程写 debug：到底读到了什么、匹配了多少、为何为 0

重要：
- 主线 main.py 需要 import 的入口：run_step4(s, ctx) -> Dict[str, Any]
- 不能再定义第二个同名 run_step4 覆盖它
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from a_top10.config import Settings


# -------------------------
# Helpers
# -------------------------
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


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _norm_code(code: Any) -> Tuple[str, str]:
    """
    标准化股票代码：
    - 返回 (ts_code_upper, code6)
    支持：
    - 002506.SZ -> ("002506.SZ", "002506")
    - 002506 -> ("", "002506")
    """
    s = _safe_str(code).upper()
    if not s:
        return "", ""
    if "." in s:
        parts = s.split(".")
        c6 = parts[0]
        c6 = "".join([ch for ch in c6 if ch.isdigit()])[:6]
        return s, c6
    # 纯 6 位
    c6 = "".join([ch for ch in s if ch.isdigit()])[:6]
    return "", c6


# -------------------------
# Core internal
# -------------------------
def _build_industry_score_map(hot_boards: pd.DataFrame) -> Dict[str, float]:
    """
    hot_boards.csv -> industry -> score(0~1)

    你现在的 hot_boards.csv 字段是：
    trade_date, industry, limit_up_count, rank

    规则（稳定 & 可解释）：
    - 以 rank 为主：score = (K - rank + 1) / K，默认 K=10（取榜单长度）
    - 若 rank 缺失，则按行号生成 rank
    """
    if hot_boards is None or hot_boards.empty:
        return {}

    ind_col = _first_existing_col(hot_boards, ["industry", "行业", "板块"])
    rank_col = _first_existing_col(hot_boards, ["rank", "rnk", "排名"])

    if not ind_col:
        return {}

    tmp = hot_boards.copy()
    tmp[ind_col] = tmp[ind_col].astype(str).map(_safe_str)
    tmp = tmp[tmp[ind_col] != ""].copy()
    if tmp.empty:
        return {}

    if rank_col:
        tmp["_rank"] = tmp[rank_col].map(lambda v: _to_int(v, 9999))
    else:
        tmp["_rank"] = np.arange(1, len(tmp) + 1)

    # 只取前 K 名更稳定
    K = int(min(10, len(tmp)))
    tmp = tmp.sort_values("_rank", ascending=True).head(K).copy()

    def rank_to_score(r: int) -> float:
        r = int(r)
        if r <= 0:
            r = 1
        return _clip01((K - r + 1) / float(K))

    tmp["_score"] = tmp["_rank"].map(rank_to_score)

    out: Dict[str, float] = {}
    for _, row in tmp.iterrows():
        ind = _safe_str(row[ind_col])
        sc = float(row["_score"])
        if ind and (ind not in out or sc > out[ind]):
            out[ind] = sc
    return out


def _extract_dragon_set(dragon: pd.DataFrame) -> set[str]:
    """
    龙虎榜命中集合：同时支持 ts_code 形态与 code6 形态
    返回集合里只存两种 key：
    - "TS:002506.SZ"
    - "C6:002506"
    """
    if dragon is None or dragon.empty:
        return set()

    code_col = _first_existing_col(dragon, ["ts_code", "code", "股票代码", "证券代码"])
    if not code_col:
        return set()

    sset: set[str] = set()
    for v in dragon[code_col].astype(str).tolist():
        ts, c6 = _norm_code(v)
        if ts:
            sset.add(f"TS:{ts}")
        if c6:
            sset.add(f"C6:{c6}")
    return sset


def _ensure_industry(df: pd.DataFrame, ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    """
    若 df 缺 industry/行业，则尝试用 ctx["stock_basic"] 补齐。
    返回 (new_df, filled_count)
    """
    if df is None or df.empty:
        return df, 0

    # df 已有行业字段就不补
    ind_col_df = _first_existing_col(df, ["industry", "行业", "所属行业", "板块"])
    if ind_col_df:
        return df, int((df[ind_col_df].astype(str).map(_safe_str) != "").sum())

    stock_basic = ctx.get("stock_basic")
    if not isinstance(stock_basic, pd.DataFrame) or stock_basic.empty:
        return df, 0

    # stock_basic 需要：ts_code + industry（或 行业）
    sb_code = _first_existing_col(stock_basic, ["ts_code", "code", "股票代码", "证券代码"])
    sb_ind = _first_existing_col(stock_basic, ["industry", "行业", "所属行业"])
    if not sb_code or not sb_ind:
        return df, 0

    out = df.copy()

    df_code = _first_existing_col(out, ["ts_code", "code", "股票代码", "证券代码"])
    if not df_code:
        return df, 0

    # 统一用 code6 做 join，避免 .SZ/.SH 不一致
    out["_code6"] = out[df_code].map(lambda x: _norm_code(x)[1])
    sb = stock_basic.copy()
    sb["_code6"] = sb[sb_code].map(lambda x: _norm_code(x)[1])
    sb = sb[sb["_code6"].astype(str).map(_safe_str) != ""].copy()
    sb["_industry"] = sb[sb_ind].astype(str).map(_safe_str)
    sb = sb[sb["_industry"] != ""].copy()

    if sb.empty:
        out.drop(columns=["_code6"], inplace=True, errors="ignore")
        return out, 0

    out = out.merge(sb[["_code6", "_industry"]].drop_duplicates("_code6"), on="_code6", how="left")
    out.rename(columns={"_industry": "industry"}, inplace=True)

    filled = int((out["industry"].astype(str).map(_safe_str) != "").sum())
    out.drop(columns=["_code6"], inplace=True, errors="ignore")
    return out, filled


def step4_theme_boost(
    strength_df: pd.DataFrame,
    ctx: Optional[Dict[str, Any]] = None,
    *,
    dragon_bonus: float = 0.08,
) -> pd.DataFrame:
    """
    输出 “题材加成”(0~1) —— 最小闭环版：
    - 行业热度：来自 ctx["hot_boards"]（即 hot_boards.csv）
    - 龙虎榜：来自 ctx["top_list"]（若命中额外加 dragon_bonus）
    - 若行业缺失：自动从 ctx["stock_basic"] 补齐 industry

    默认：
    - 没匹配到行业热度 -> 题材加成 = 0
    - 匹配到 -> 题材加成 = industry_score(0~1)
    - 龙虎榜命中 -> + dragon_bonus 再 clip(0~1)
    """
    if strength_df is None or strength_df.empty:
        return pd.DataFrame()

    ctx = ctx or {}
    df = strength_df.copy()

    # 题材源
    hot_boards_df = ctx.get("hot_boards")
    if not isinstance(hot_boards_df, pd.DataFrame):
        hot_boards_df = ctx.get("boards") if isinstance(ctx.get("boards"), pd.DataFrame) else None

    dragon_df = ctx.get("top_list")
    if not isinstance(dragon_df, pd.DataFrame):
        dragon_df = ctx.get("dragon") if isinstance(ctx.get("dragon"), pd.DataFrame) else None

    industry_score_map = _build_industry_score_map(hot_boards_df) if isinstance(hot_boards_df, pd.DataFrame) else {}
    dragon_set = _extract_dragon_set(dragon_df) if isinstance(dragon_df, pd.DataFrame) else set()

    # 先保证 df 有 industry
    df, industry_filled = _ensure_industry(df, ctx)

    # 初始化输出列（保证 writer 一定能写出来）
    df["题材加成"] = 0.0
    df["板块"] = ""
    df["_ThemeIndustryHit"] = 0
    df["_ThemeDragonHit"] = 0

    ind_col = _first_existing_col(df, ["industry", "行业", "所属行业", "板块"])
    code_col = _first_existing_col(df, ["ts_code", "code", "股票代码", "证券代码"])

    # 行业热度加成
    matched_industry = 0
    if industry_score_map and ind_col:
        for i in range(len(df)):
            ind = _safe_str(df.iloc[i].get(ind_col))
            if ind:
                df.at[df.index[i], "板块"] = ind
                sc = industry_score_map.get(ind)
                if sc is not None:
                    matched_industry += 1
                    df.at[df.index[i], "_ThemeIndustryHit"] = 1
                    df.at[df.index[i], "题材加成"] = float(_clip01(sc))

    # 龙虎榜额外加成（支持 ts_code / code6）
    dragon_hits = 0
    if dragon_set and code_col:
        for i in range(len(df)):
            ts, c6 = _norm_code(df.iloc[i].get(code_col))
            hit = False
            if ts and f"TS:{ts}" in dragon_set:
                hit = True
            elif c6 and f"C6:{c6}" in dragon_set:
                hit = True

            if hit:
                dragon_hits += 1
                df.at[df.index[i], "_ThemeDragonHit"] = 1
                df.at[df.index[i], "题材加成"] = float(_clip01(float(df.at[df.index[i], "题材加成"]) + float(dragon_bonus)))

    # 最终 clip
    df["题材加成"] = df["题材加成"].map(lambda x: float(_clip01(_to_float(x, 0.0))))

    # 把 step4 关键诊断写入 ctx.debug（若存在）
    dbg = ctx.get("debug")
    if isinstance(dbg, dict):
        dbg.setdefault("step4_theme", {})
        dbg["step4_theme"].update(
            {
                "hot_boards_rows": int(len(hot_boards_df)) if isinstance(hot_boards_df, pd.DataFrame) else 0,
                "hot_boards_cols": list(hot_boards_df.columns) if isinstance(hot_boards_df, pd.DataFrame) else [],
                "industry_score_map_size": int(len(industry_score_map)),
                "industry_filled_count": int(industry_filled),
                "matched_industry_count": int(matched_industry),
                "dragon_set_size": int(len(dragon_set)),
                "dragon_hits": int(dragon_hits),
                "note": "题材加成=行业热度(0~1) + 龙虎榜加成；缺行业则从 stock_basic 补齐",
            }
        )

    return df


# -------------------------
# Mainline entrypoint (required by a_top10.main import)
# -------------------------
def run_step4(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    主线入口：从 ctx 中找到 step3 输出的 df，做题材加权，再写回 ctx。
    兼容 key（不写死）：优先按这些顺序找：
    - ctx["strength_df"] / ctx["strength"]
    - ctx["step3"] / ctx["step3_df"]
    - ctx["candidates"] / ctx["pool"]（找不到也不报错）
    """
    if not isinstance(ctx, dict):
        return {"debug": {"step4": {"status": "skip", "note": "ctx 非 dict"}}}

    dbg = ctx.get("debug")
    if not isinstance(dbg, dict):
        dbg = {}
        ctx["debug"] = dbg

    info: Dict[str, Any] = {"step": "step4_theme_boost", "status": "ok", "note": ""}

    keys_try = ["strength_df", "strength", "step3", "step3_df", "candidates", "pool"]
    src_key = None
    src_df = None
    for k in keys_try:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            src_key = k
            src_df = v
            break

    if src_df is None or src_df.empty:
        info["status"] = "skip"
        info["note"] = "ctx 中未找到可用的 DataFrame（strength/step3/candidates/pool），step4 跳过"
        dbg["step4"] = info
        return ctx

    out = step4_theme_boost(src_df, ctx)

    # 写回同一个 key，保证后续步骤读到的是加权后的 df
    ctx[src_key] = out
    info["source_key"] = src_key
    info["rows"] = int(len(out))

    # 额外写一点关键统计，方便你一眼判断“是不是全 0”
    try:
        info["theme_nonzero"] = int((pd.to_numeric(out["题材加成"], errors="coerce").fillna(0.0) > 0).sum())
    except Exception:
        info["theme_nonzero"] = -1

    dbg["step4"] = info
    return ctx


# 便捷函数：给你本地/旁路调试用，不影响主线 import
def run_step4_df(df: pd.DataFrame, ctx: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return step4_theme_boost(df, ctx)
