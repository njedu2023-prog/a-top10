#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step4 : 题材/板块加权（ThemeBoost）

输入：
    strength_df（step3输出）
    + 可选：s（Settings/ctx），用于拿 step0 的 hot_boards / 龙虎榜等

输出：
    theme_df（附带 ThemeBoost + 分项）
"""

from __future__ import annotations

from typing import Optional, Sequence

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


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float) -> pd.Series:
    if (col is None) or (col not in df.columns):
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


def _ensure_ts_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts_col = _first_existing_col(out, ["ts_code", "code", "TS_CODE"])
    if ts_col is None:
        out["ts_code"] = ""
    else:
        out["ts_code"] = out[ts_col].astype(str)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------
# Step0 context fetch (boards / dragon)
# -------------------------
def _get_step0_boards(s) -> pd.DataFrame:
    """
    兼容：s 可能是 Settings，也可能是 ctx(dict)
    期望得到 hot_boards DataFrame（包含 板块名 + 热度排名/热度值）
    """
    if s is None:
        return pd.DataFrame()

    # ctx(dict) 情况
    if isinstance(s, dict):
        for k in ["boards", "hot_boards", "hot_board_df"]:
            v = s.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame()

    # Settings 情况：有些项目会把 ctx 放在 s.ctx / s.context
    for attr in ["ctx", "context"]:
        if hasattr(s, attr):
            ctx = getattr(s, attr)
            if isinstance(ctx, dict):
                for k in ["boards", "hot_boards", "hot_board_df"]:
                    v = ctx.get(k, None)
                    if isinstance(v, pd.DataFrame):
                        return v

    return pd.DataFrame()


def _get_step0_dragon(s) -> pd.DataFrame:
    """
    兼容：龙虎榜/龙头数据
    - step0 里你叫 top_list，并放到了 ctx["dragon"]
    """
    if s is None:
        return pd.DataFrame()

    if isinstance(s, dict):
        for k in ["dragon", "top_list", "dragon_df"]:
            v = s.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame()

    for attr in ["ctx", "context"]:
        if hasattr(s, attr):
            ctx = getattr(s, attr)
            if isinstance(ctx, dict):
                for k in ["dragon", "top_list", "dragon_df"]:
                    v = ctx.get(k, None)
                    if isinstance(v, pd.DataFrame):
                        return v

    return pd.DataFrame()


# -------------------------
# Theme Boost Core
# -------------------------
def calc_theme_boost(strength_df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    ThemeBoost = 0.8 ~ 1.3 之间的平滑加权（sigmoid）
    - 板块热度（rank越小越热） -> _score_board
    - 龙虎榜/龙头奖励 -> _score_dragon
    """
    if strength_df is None or len(strength_df) == 0:
        return pd.DataFrame(columns=["ThemeBoost"])

    out = _ensure_ts_code(strength_df)

    # --- 1) 合并板块热度 ---
    boards = _get_step0_boards(s)
    board_hot_rank = pd.Series([50.0] * len(out), index=out.index, dtype="float64")  # 默认不热

    # strength_df 里可能有板块字段
    stock_board_col = _first_existing_col(out, ["board", "concept", "theme", "industry", "板块", "概念"])
    if (not boards.empty) and (stock_board_col is not None):
        b = boards.copy()

        # boards 里板块名字段
        b_name_col = _first_existing_col(b, ["board", "name", "concept", "theme", "industry", "板块", "概念"])
        # boards 里热度排名字段（没有就用热度值反推）
        b_rank_col = _first_existing_col(b, ["hot_rank", "rank", "board_hot_rank", "热度排名", "排名"])
        b_hot_col = _first_existing_col(b, ["hot", "heat", "score", "热度", "热度值"])

        if b_name_col is not None:
            b["_board_name"] = b[b_name_col].astype(str)

            if b_rank_col is not None:
                b["_hot_rank"] = pd.to_numeric(b[b_rank_col], errors="coerce")
            elif b_hot_col is not None:
                # 用热度值反推排名：热度越大排名越靠前
                hot = pd.to_numeric(b[b_hot_col], errors="coerce").fillna(0.0)
                b["_hot_rank"] = hot.rank(ascending=False, method="min")
            else:
                b["_hot_rank"] = np.nan

            b = b[["_board_name", "_hot_rank"]].dropna()
            if not b.empty:
                out["_board_name"] = out[stock_board_col].astype(str)
                out = out.merge(b, how="left", left_on="_board_name", right_on="_board_name")
                board_hot_rank = pd.to_numeric(out["_hot_rank"], errors="coerce").fillna(50.0).astype("float64")
                out.drop(columns=[c for c in ["_hot_rank"] if c in out.columns], inplace=True)

    out["board_hot_rank"] = board_hot_rank.clip(1.0, 200.0)

    # --- 2) 龙虎榜/龙头奖励 ---
    dragon = _get_step0_dragon(s)
    dragon_flag = pd.Series([0.0] * len(out), index=out.index, dtype="float64")

    if not dragon.empty:
        ts_col = _first_existing_col(dragon, ["ts_code", "code", "TS_CODE"])
        if ts_col is not None:
            dragon_set = set(dragon[ts_col].astype(str).tolist())
            dragon_flag = out["ts_code"].astype(str).apply(lambda x: 1.0 if x in dragon_set else 0.0).astype("float64")

    out["dragon_flag"] = dragon_flag.clip(0.0, 1.0)

    # --- 3) 分项打分 ---
    # rank 越小越热：1名接近1分，50名接近0分
    out["_score_board"] = (1.0 - (out["board_hot_rank"] / 50.0)).clip(0.0, 1.0)
    out["_score_dragon"] = out["dragon_flag"].clip(0.0, 1.0)

    # --- 4) Sigmoid 合成（0.8~1.3） ---
    raw = out["_score_board"].astype(float).values + 0.6 * out["_score_dragon"].astype(float).values
    out["ThemeBoost"] = 0.8 + 0.5 * _sigmoid(3.0 * (raw - 0.5))

    return out.sort_values("ThemeBoost", ascending=False)


# -------------------------
# Runner
# -------------------------
def run_step4(strength_df: pd.DataFrame, s=None) -> pd.DataFrame:
    return calc_theme_boost(strength_df, s=s)


# Backward-compatible alias
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step4(df, s=s)


if __name__ == "__main__":
    print("Step4 ThemeBoost ready.")
