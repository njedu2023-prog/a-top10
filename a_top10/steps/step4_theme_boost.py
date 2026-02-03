#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step4 : 题材/板块加权（ThemeBoost）

输入：
    strength_df（step3输出）
    hot_board_df（step0输入热板块）

输出：
    theme_df（附带 ThemeBoost）
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# -------------------------
# Theme Boost Core
# -------------------------

def calc_theme_boost(df: pd.DataFrame) -> pd.DataFrame:
    """
    ThemeBoost = sigmoid(概念热度 + 龙头位置)
    """

    out = df.copy()

    # 字段约定：
    # board_hot_rank：板块热度排名（1最热）
    # dragon_flag：是否龙头（1/0）

    if "board_hot_rank" not in out.columns:
        out["board_hot_rank"] = 50

    if "dragon_flag" not in out.columns:
        out["dragon_flag"] = 0

    # 热度评分：rank越小越强
    out["score_board"] = 1.0 - (out["board_hot_rank"] / 50).clip(0, 1)

    # 龙头奖励
    out["score_dragon"] = out["dragon_flag"].clip(0, 1)

    # Sigmoid 合成（0.8~1.3）
    raw = out["score_board"] + 0.6 * out["score_dragon"]

    out["ThemeBoost"] = 0.8 + 0.5 * (1 / (1 + np.exp(-3 * (raw - 0.5))))

    return out.sort_values("ThemeBoost", ascending=False)


# -------------------------
# Runner
# -------------------------

def run_step4(strength_df: pd.DataFrame) -> pd.DataFrame:
    return calc_theme_boost(strength_df)


if __name__ == "__main__":
    print("Step4 ThemeBoost ready.")
