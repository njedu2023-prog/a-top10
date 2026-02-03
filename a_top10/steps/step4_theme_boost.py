#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step4 : 题材/板块加权（ThemeBoost）

输入：
    strength_df（step3输出）
    hot_boards_df（step0输入热板块，可选）

输出：
    theme_df（附带 ThemeBoost + 分项解释）

设计目标：
- 能对接真实 hot_boards.csv
- 字段缺失可运行（兜底）
- 输出可解释（score_board/score_dragon/raw_theme）
- 主程序统一调用兼容 run()
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return first existing column name (case-insensitive)."""
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
    """Convert column safely to float."""
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


# -------------------------
# Theme Boost Core
# -------------------------

def calc_theme_boost(df: pd.DataFrame) -> pd.DataFrame:
    """
    ThemeBoost = 0.8 ~ 1.3
    综合：
      - 板块热度 rank 越高越强
      - 龙头标记 dragon_flag 加分
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ThemeBoost"])

    out = df.copy()

    # ========= 板块热度字段 =========
    rank_col = _first_existing_col(
        out,
        ["board_hot_rank", "hot_rank", "rank", "concept_rank", "题材热度排名"],
    )
    board_hot_rank = _to_float_series(out, rank_col, default=50.0)
    board_hot_rank = board_hot_rank.clip(1.0, 50.0)

    out["board_hot_rank"] = board_hot_rank

    # 热度评分：rank越小越强
    out["score_board"] = 1.0 - (board_hot_rank / 50.0).clip(0, 1)

    # ========= 龙头奖励字段 =========
    dragon_col = _first_existing_col(
        out,
        ["dragon_flag", "is_dragon", "core_flag", "龙头标记"],
    )
    dragon_flag = _to_float_series(out, dragon_col, default=0.0)
    dragon_flag = dragon_flag.clip(0.0, 1.0)

    out["dragon_flag"] = dragon_flag
    out["score_dragon"] = dragon_flag

    # ========= Sigmoid 合成 =========
    raw = out["score_board"] + 0.6 * out["score_dragon"]
    out["raw_theme"] = raw

    # 输出 ThemeBoost：范围 0.8 ~ 1.3
    out["ThemeBoost"] = 0.8 + 0.5 * (1 / (1 + np.exp(-3 * (raw - 0.5))))

    return out.sort_values("ThemeBoost", ascending=False)


# -------------------------
# Runner
# -------------------------

def run_step4(strength_df: pd.DataFrame, s=None) -> pd.DataFrame:
    return calc_theme_boost(strength_df)


# 主程序统一接口兼容
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step4(df, s=s)


if __name__ == "__main__":
    print("Step4 ThemeBoost module loaded successfully.")
