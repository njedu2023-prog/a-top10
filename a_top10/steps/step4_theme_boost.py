#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4 : 题材/板块加权（ThemeBoost）

- 你主线 main.py 里 import 的是：run_step4
- 你文件里原本只有：step4_theme_boost
=> 本文件补齐 run_step4 入口，避免 ImportError，同时不破坏你现有逻辑
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

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


# -------------------------
# Core internal
# -------------------------
def _build_board_score_map(boards: pd.DataFrame) -> Dict[str, float]:
    """
    将 hot_boards 表转换成：industry -> board_score(0~1)
    规则：综合 rank（越小越好） + limit_up_count（越大越好）
    """
    if boards is None or boards.empty:
        return {}

    ind_col = _first_existing_col(boards, ["industry", "board", "sector", "板块", "行业"])
    cnt_col = _first_existing_col(boards, ["limit_up_count", "limitup_count", "cnt", "count", "涨停数"])
    rank_col = _first_existing_col(boards, ["rank", "rnk", "排名"])

    if not ind_col:
        return {}

    tmp = boards.copy()
    tmp[ind_col] = tmp[ind_col].astype(str).map(_safe_str)
    tmp = tmp[tmp[ind_col] != ""].copy()
    if tmp.empty:
        return {}

    if rank_col:
        tmp["_rank"] = tmp[rank_col].map(lambda v: _to_int(v, 9999))
    else:
        tmp["_rank"] = np.arange(1, len(tmp) + 1)

    if cnt_col:
        tmp["_cnt"] = tmp[cnt_col].map(lambda v: _to_int(v, 0))
    else:
        tmp["_cnt"] = 0

    max_rank = max(10, int(tmp["_rank"].max() if len(tmp) else 10))
    denom_rank = (max_rank - 1) if max_rank > 1 else 1
    tmp["_rank_score"] = tmp["_rank"].map(lambda r: _clip01(1.0 - (r - 1) / denom_rank))

    cmin = float(tmp["_cnt"].min())
    cmax = float(tmp["_cnt"].max())
    if cmax <= cmin:
        tmp["_cnt_score"] = 0.5
    else:
        tmp["_cnt_score"] = (tmp["_cnt"] - cmin) / (cmax - cmin)

    tmp["_board_score"] = (0.60 * tmp["_rank_score"] + 0.40 * tmp["_cnt_score"]).map(_clip01)

    out: Dict[str, float] = {}
    for _, row in tmp.iterrows():
        ind = _safe_str(row[ind_col])
        sc = float(row["_board_score"])
        if ind and (ind not in out or sc > out[ind]):
            out[ind] = sc
    return out


def _extract_dragon_set(dragon: pd.DataFrame) -> set[str]:
    if dragon is None or dragon.empty:
        return set()

    code_col = _first_existing_col(dragon, ["ts_code", "code", "股票代码", "证券代码"])
    if not code_col:
        return set()

    s: set[str] = set()
    for v in dragon[code_col].astype(str).tolist():
        k = _safe_str(v).upper()
        if k:
            s.add(k)
    return s


def _infer_stock_board(row: pd.Series) -> str:
    for cands in [
        ["板块", "行业", "industry", "board", "sector", "concept"],
        ["所属行业", "所属板块"],
    ]:
        for name in cands:
            if name in row.index:
                v = _safe_str(row.get(name))
                if v:
                    return v
    return ""


def step4_theme_boost(
    strength_df: pd.DataFrame,
    ctx: Optional[Dict[str, Any]] = None,
    *,
    default_theme: float = 0.891213,
    min_theme: float = 0.82,
    max_theme: float = 0.96,
    dragon_bonus: float = 0.015,
) -> pd.DataFrame:
    """
    输出 “题材加成”(0~1)：
    - 默认给 default_theme
    - 若能识别到板块热度，则按板块热度做小幅上下浮动
    - 若在龙虎榜，则额外 + dragon_bonus（再 clip）
    """
    if strength_df is None or strength_df.empty:
        return pd.DataFrame()

    df = strength_df.copy()

    ctx = ctx or {}
    boards_df = ctx.get("boards", None) or ctx.get("hot_boards", None)
    dragon_df = ctx.get("dragon", None) or ctx.get("top_list", None)

    board_score_map = _build_board_score_map(boards_df) if isinstance(boards_df, pd.DataFrame) else {}
    dragon_set = _extract_dragon_set(dragon_df) if isinstance(dragon_df, pd.DataFrame) else set()

    code_col = _first_existing_col(df, ["ts_code", "code", "股票代码", "证券代码"])

    df["题材加成"] = float(default_theme)
    df["板块"] = ""
    df["_ThemeBoardScore"] = 0.0
    df["_ThemeDragonHit"] = 0

    if board_score_map:
        def map_theme(bs: float) -> float:
            bs = _clip01(float(bs))
            return float(min_theme + (max_theme - min_theme) * bs)

        for i in range(len(df)):
            row = df.iloc[i]
            board = _infer_stock_board(row)
            if board:
                df.at[df.index[i], "板块"] = board
                bs = board_score_map.get(board, None)
                if bs is not None:
                    df.at[df.index[i], "_ThemeBoardScore"] = float(bs)
                    df.at[df.index[i], "题材加成"] = map_theme(float(bs))

    if dragon_set and code_col:
        for i in range(len(df)):
            code = _safe_str(df.iloc[i].get(code_col)).upper()
            if code and code in dragon_set:
                df.at[df.index[i], "_ThemeDragonHit"] = 1
                df.at[df.index[i], "题材加成"] = float(df.at[df.index[i], "题材加成"]) + float(dragon_bonus)

    df["题材加成"] = df["题材加成"].map(
        lambda x: float(max(min_theme, min(max_theme, _to_float(x, default_theme))))
    )

    return df


# -------------------------
# Mainline entrypoint (required by a_top10.main import)
# -------------------------
def run_step4(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    主线入口：从 ctx 中找到 step3 输出的 strength_df，做题材加权，再写回 ctx

    兼容 key（不写死）：优先按这些顺序找
    - ctx["strength_df"] / ctx["strength"]
    - ctx["step3"] / ctx["step3_df"]
    - ctx["candidates"] / ctx["pool"]（实在找不到 step3 输出时也不报错）
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
    dbg["step4"] = info
    return ctx
    
def run_step4(df: pd.DataFrame, ctx: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return step4_theme_boost(df, ctx)
