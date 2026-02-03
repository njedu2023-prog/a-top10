#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: 涨停质量评分（A+B+C 手工强信号）

输入：
    candidates_df（来自 step2），至少应包含 ts_code / name（不强制）

输出：
    strength_df（附带 StrengthScore + 各分项）
    默认筛选前 Top50（可配置 top_k）

设计目标：
- 字段缺失可运行（兜底）
- 字段类型混乱可运行（字符串/空值/inf -> 自动清洗）
- 评分可解释（输出分项 score_*）
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


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(0.0, 1.0)


# -------------------------
# Strength Score Core
# -------------------------

def calc_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    StrengthScore = 0.35*封板早 + 0.25*炸板少 + 0.20*封单大 + 0.20*换手合理

    字段兼容（尽量对接你们的数据仓库口径）：
    - 封板时间：first_limit_time / first_hit_limit_time / 封板时间
      口径：分钟序号(0~240)，越小越早越强
    - 炸板次数：open_times / break_times / 炸板次数
      口径：次数，越少越强
    - 封单金额：seal_amount / seal_amt / 封单金额
      口径：元，越大越强（按 1e8 归一）
    - 换手率：turnover_rate / turn_rate / 换手率
      口径：百分比（5~15 最优，过高过低扣分）
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["StrengthScore"])

    out = df.copy()

    # ========= A1 封板时间（越早越强） =========
    time_col = _first_existing_col(out, ["first_limit_time", "first_hit_limit_time", "封板时间", "first_limit_min"])
    first_limit_time = _to_float_series(out, time_col, default=240.0)
    # 合理区间 0~240，越小越强
    first_limit_time = first_limit_time.clip(0.0, 240.0)
    out["first_limit_time"] = first_limit_time
    out["score_time"] = _clip01(1.0 - (first_limit_time / 240.0))

    # ========= A3 炸板次数（越少越强） =========
    open_col = _first_existing_col(out, ["open_times", "break_times", "炸板次数", "open_cnt"])
    open_times = _to_float_series(out, open_col, default=3.0)
    open_times = open_times.clip(0.0, 20.0)
    out["open_times"] = open_times
    # 0~5 次线性扣分，>5 视为0分
    out["score_open"] = _clip01(1.0 - (open_times / 5.0))

    # ========= A5 封单金额（越大越强） =========
    seal_col = _first_existing_col(out, ["seal_amount", "seal_amt", "封单金额", "seal_money"])
    seal_amount = _to_float_series(out, seal_col, default=0.0)
    seal_amount = seal_amount.clip(0.0, float("inf"))
    out["seal_amount"] = seal_amount
    # 1e8（1亿）封单金额视为满分
    out["score_seal"] = _clip01(seal_amount / 1e8)

    # ========= 换手率合理（过高过低都扣分） =========
    tr_col = _first_existing_col(out, ["turnover_rate", "turn_rate", "换手率", "turnover"])
    turnover_rate = _to_float_series(out, tr_col, default=5.0)
    turnover_rate = turnover_rate.clip(0.0, 100.0)
    out["turnover_rate"] = turnover_rate

    # 最优换手区间 5%~15%，中心 10%，偏离越多扣分
    out["score_turnover"] = _clip01(1.0 - ((turnover_rate - 10.0).abs() / 10.0))

    # ========= 综合 StrengthScore =========
    out["StrengthScore"] = (
        0.35 * out["score_time"]
        + 0.25 * out["score_open"]
        + 0.20 * out["score_seal"]
        + 0.20 * out["score_turnover"]
    ) * 100.0

    return out.sort_values("StrengthScore", ascending=False)


# -------------------------
# Main Runner
# -------------------------

def run_step3(candidates_df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """
    返回 StrengthScore TopK 股票进入下一层
    """
    scored = calc_strength_score(candidates_df)
    top_k = int(top_k or 50)
    top_k = max(1, top_k)

    if len(scored) > top_k:
        scored = scored.head(top_k)

    return scored


# Backward-compatible alias：主程序若统一用 run() 调用每一步，这里也给出
def run(df: pd.DataFrame, s=None, top_k: int = 50) -> pd.DataFrame:
    return run_step3(df, top_k=top_k)


if __name__ == "__main__":
    print("Step3 module loaded successfully.")
