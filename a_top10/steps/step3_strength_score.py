#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3 : 涨停质量评分（A+B+C 手工强信号）

输入：
    candidates_df（来自 step2）

输出：
    strength_df（附带 StrengthScore）
    默认筛选前 Top50
"""

from __future__ import annotations

import pandas as pd


# -------------------------
# Strength Score Core
# -------------------------

def calc_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    StrengthScore = 0.35*封板早 + 0.25*炸板少 + 0.20*封单大 + 0.20*换手合理
    """

    out = df.copy()

    # ========= A1 封板时间（越早越强） =========
    # 字段约定：first_limit_time（分钟序号越小越早）
    if "first_limit_time" not in out.columns:
        out["first_limit_time"] = 240  # fallback：默认尾盘

    out["score_time"] = 1.0 - (out["first_limit_time"] / 240).clip(0, 1)

    # ========= A3 炸板次数（越少越强） =========
    if "open_times" not in out.columns:
        out["open_times"] = 3

    out["score_open"] = 1.0 - (out["open_times"] / 5).clip(0, 1)

    # ========= A5 封单金额（越大越强） =========
    if "seal_amount" not in out.columns:
        out["seal_amount"] = 0

    out["score_seal"] = (out["seal_amount"] / 1e8).clip(0, 1)

    # ========= 换手率合理（过高过低都扣分） =========
    if "turnover_rate" not in out.columns:
        out["turnover_rate"] = 5

    # 最优换手区间 5%~15%
    tr = out["turnover_rate"]
    out["score_turnover"] = 1.0 - ((tr - 10).abs() / 10).clip(0, 1)

    # ========= 综合 StrengthScore =========
    out["StrengthScore"] = (
        0.35 * out["score_time"]
        + 0.25 * out["score_open"]
        + 0.20 * out["score_seal"]
        + 0.20 * out["score_turnover"]
    ) * 100

    return out.sort_values("StrengthScore", ascending=False)


# -------------------------
# Main Runner
# -------------------------

def run_step3(candidates_df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """
    返回 StrengthScore TopK 股票进入下一层
    """
    scored = calc_strength_score(candidates_df)

    if len(scored) > top_k:
        scored = scored.head(top_k)

    return scored


if __name__ == "__main__":
    print("Step3 module loaded successfully.")
