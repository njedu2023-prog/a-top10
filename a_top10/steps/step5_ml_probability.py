#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : 概率模型推断（ML核心层）

输入：
    theme_df（step4输出）

输出：
    prob_df（附带 Probability）

模型：
    LogisticRegression baseline
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# -------------------------
# Feature Columns (真实字段)
# -------------------------

FEATURES = [
    "StrengthScore",
    "ThemeBoost",
    "seal_amount",
    "open_times",
    "turnover_rate",
]


# -------------------------
# Probability Model
# -------------------------

def run_step5(theme_df: pd.DataFrame) -> pd.DataFrame:
    """
    轻量 Logistic 推断：
    输出 Probability（明日涨停概率）
    """

    out = theme_df.copy()

    # 缺失补0
    for c in FEATURES:
        if c not in out.columns:
            out[c] = 0

    X = out[FEATURES].fillna(0)

    # ⚠️ 目前无训练标签，只能先用规则初始化一个 pseudo-model
    # 后续 Step7 自学习闭环替换为真实训练

    # pseudo probability (sigmoid)
    z = (
        0.04 * out["StrengthScore"]
        + 1.5 * out["ThemeBoost"]
        - 0.3 * out["open_times"]
    )

    out["Probability"] = 1 / (1 + np.exp(-z))

    return out.sort_values("Probability", ascending=False)


if __name__ == "__main__":
    print("Step5 Probability model loaded.")
