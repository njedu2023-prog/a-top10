#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4 : 题材/板块加权（ThemeBoost）

输入：
- strength_df（step3 输出，至少含 ts_code / name 其一）
- ctx（step0 输出，建议包含）
    ctx["boards"] : hot_boards.csv 读入后的 DataFrame
        期望列：industry, limit_up_count, rank（列名大小写不敏感）
    ctx["dragon"] : 龙虎榜 DataFrame（如存在）
        期望列：ts_code（或类似可识别列）

输出：
- theme_df：在 strength_df 基础上新增/覆盖：
    - "题材加成"  (float)  —— 0~1 之间的加成因子（越大越有利）
    - "板块"      (str)    —— 识别到的板块/行业（如果能识别）
  以及若干调试分项列（不会影响后续也可以不用）
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    s = str(x)
    return s.strip()


# -------------------------
# Core
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

    # rank 越小越好；如果没有 rank，就按出现顺序给 rank
    if rank_col:
        tmp["_rank"] = tmp[rank_col].map(lambda v: _to_int(v, 9999))
    else:
        tmp["_rank"] = np.arange(1, len(tmp) + 1)

    # 涨停数越大越好；如果没有就默认为 0
    if cnt_col:
        tmp["_cnt"] = tmp[cnt_col].map(lambda v: _to_int(v, 0))
    else:
        tmp["_cnt"] = 0

    # 归一化
    # rank_score：rank=1 -> 1.0；rank=10 -> 0.0（线性）
    max_rank = max(10, int(tmp["_rank"].max() if len(tmp) else 10))
    tmp["_rank_score"] = tmp["_rank"].map(lambda r: _clip01(1.0 - (r - 1) / (max_rank - 1 if max_rank > 1 else 1)))

    # cnt_score：用 min-max（避免极端值）
    cmin = float(tmp["_cnt"].min())
    cmax = float(tmp["_cnt"].max())
    if cmax <= cmin:
        tmp["_cnt_score"] = 0.5  # 全一样就给中性
    else:
        tmp["_cnt_score"] = (tmp["_cnt"] - cmin) / (cmax - cmin)

    # 合成
    # rank 权重略大（更稳），cnt 次之（更敏感）
    tmp["_board_score"] = (0.60 * tmp["_rank_score"] + 0.40 * tmp["_cnt_score"]).map(_clip01)

    # 去重：同一 industry 取最大分
    out: Dict[str, float] = {}
    for _, row in tmp.iterrows():
        ind = _safe_str(row[ind_col])
        sc = float(row["_board_score"])
        if ind and (ind not in out or sc > out[ind]):
            out[ind] = sc
    return out


def _extract_dragon_set(dragon: pd.DataFrame) -> set[str]:
    """
    龙虎榜集合：ts_code（大小写/空格做一下清洗）
    """
    if dragon is None or dragon.empty:
        return set()

    code_col = _first_existing_col(dragon, ["ts_code", "code", "股票代码", "证券代码"])
    if not code_col:
        return set()

    s = set()
    for v in dragon[code_col].astype(str).tolist():
        k = _safe_str(v).upper()
        if k:
            s.add(k)
    return s


def _infer_stock_board(row: pd.Series) -> str:
    """
    尝试从 strength_df 中推断板块字段（如果你 step3 已经带了）
    """
    for cands in [
        ["板块", "行业", "industry", "board", "sector", "concept"],
        ["所属行业", "所属板块"],
    ]:
        col = None
        for name in cands:
            if name in row.index:
                col = name
                break
        if col:
            v = _safe_str(row.get(col))
            if v:
                return v
    return ""


def step4_theme_boost(
    strength_df: pd.DataFrame,
    ctx: Optional[Dict[str, Any]] = None,
    *,
    default_theme: float = 0.891213,
    # 下面参数你后面想调再调，不调也不会影响别的步骤
    min_theme: float = 0.82,
    max_theme: float = 0.96,
    dragon_bonus: float = 0.015,
) -> pd.DataFrame:
    """
    输出 “题材加成”(0~1)：
    - 默认给 default_theme（保持稳定，避免影响其它东西）
    - 若能识别到板块热度，则按板块热度对 default_theme 做小幅上下浮动
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

    # 找 ts_code 列（没有也不报错）
    code_col = _first_existing_col(df, ["ts_code", "code", "股票代码", "证券代码"])
    name_col = _first_existing_col(df, ["name", "名称", "股票名称"])

    # 题材加成初始化（稳定默认值）
    df["题材加成"] = float(default_theme)
    df["板块"] = ""

    # 分项（可选调试列）
    df["_ThemeBoardScore"] = 0.0
    df["_ThemeDragonHit"] = 0

    if board_score_map:
        # 让 theme 在 [min_theme, max_theme] 内随 board_score(0~1) 变化
        # board_score=0.5 -> 约等于 default_theme
        # board_score=1.0 -> 靠近 max_theme
        # board_score=0.0 -> 靠近 min_theme
        def map_theme(bs: float) -> float:
            bs = _clip01(float(bs))
            # 线性映射到 [min_theme, max_theme]
            return float(min_theme + (max_theme - min_theme) * bs)

        # 如果 df 自带板块字段优先用；否则就只能留空（不会影响稳定性）
        for i in range(len(df)):
            row = df.iloc[i]
            board = _infer_stock_board(row)
            if board:
                df.at[df.index[i], "板块"] = board
                bs = board_score_map.get(board, None)
                if bs is not None:
                    df.at[df.index[i], "_ThemeBoardScore"] = float(bs)
                    df.at[df.index[i], "题材加成"] = map_theme(float(bs))
            # 若没板块，不强行匹配（避免误配导致排序大波动）

    # 龙虎榜加成（小幅）
    if dragon_set and code_col:
        for i in range(len(df)):
            code = _safe_str(df.iloc[i].get(code_col)).upper()
            if code and code in dragon_set:
                df.at[df.index[i], "_ThemeDragonHit"] = 1
                df.at[df.index[i], "题材加成"] = float(df.at[df.index[i], "题材加成"]) + float(dragon_bonus)

    # 最终裁剪，避免影响过大
    df["题材加成"] = df["题材加成"].map(lambda x: float(max(min_theme, min(max_theme, _to_float(x, default_theme)))))

    # 清理一下展示（你不想要分项列也可以删）
    # 保留：题材加成、板块
    return df
