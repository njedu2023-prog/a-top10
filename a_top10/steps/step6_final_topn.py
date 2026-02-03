#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Final TopN Selector
汇总前 5 步结果：Probability × Strength × ThemeBoost × ST_penalty
得到最终打分表，并输出 TopN

输入:
    df（来自 Step5 输出）
输出:
    final_df：包含 ts_code / name / score / prob / board 等可读字段
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """返回 df 中第一个存在的列名（大小写兼容）"""
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float) -> pd.Series:
    """将列转为 float，支持 NaN / inf 兜底"""
    if (col is None) or (col not in df.columns):
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


# -------------------------
# Step6 Core
# -------------------------
def run_step6_final_topn(df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    主逻辑：
    final score = Probability × StrengthScore × ThemeBoost × ST_penalty

    输出字段：
        ts_code, name, score, prob, board
        StrengthScore, ThemeBoost（便于回测查看）
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts_code", "name", "score"])

    out_df = df.copy()

    # ---- Locate key columns ----
    ts_col = _first_existing_col(out_df, ["ts_code", "TS_CODE", "code"])
    name_col = _first_existing_col(out_df, ["name", "NAME"])
    board_col = _first_existing_col(out_df, ["board", "板块", "BOARD"])

    if ts_col is None:
        out_df["ts_code"] = ""
        ts_col = "ts_code"
    if name_col is None:
        out_df["name"] = ""
        name_col = "name"
    if board_col is None:
        out_df["board"] = ""
        board_col = "board"

    # ---- Extract core numeric features ----
    prob = _to_float_series(out_df, _first_existing_col(out_df, ["Probability", "prob", "_prob"]), 0.0)
    strength = _to_float_series(out_df, _first_existing_col(out_df, ["StrengthScore", "strength"]), 0.0)
    theme_boost = _to_float_series(out_df, _first_existing_col(out_df, ["ThemeBoost", "theme_boost"]), 1.0)

    # ThemeBoost sanity clamp
    theme_boost = theme_boost.clip(0.0, 10.0).fillna(1.0)

    # ---- ST penalty (轻惩罚，不硬拒绝) ----
    st_col = _first_existing_col(out_df, ["is_st", "st", "ST"])
    if st_col:
        is_st = _to_float_series(out_df, st_col, 0.0)
        st_penalty = np.where(is_st > 0.5, 0.85, 1.0)
    else:
        st_penalty = 1.0

    # ---- Final Score ----
    score = prob * strength * theme_boost
    if isinstance(st_penalty, np.ndarray):
        score = score * st_penalty

    out_df["_prob"] = prob
    out_df["_strength"] = strength
    out_df["_theme_boost"] = theme_boost
    out_df["_score"] = score
    out_df["_board"] = out_df[board_col].astype(str)

    # ---- Select Top N ----
    topn = 10
    if s is not None and hasattr(s, "topn"):
        try:
            topn = max(1, int(s.topn))
        except Exception:
            pass
    elif s is not None and hasattr(s, "step6"):
        try:
            topn = max(1, int(getattr(s.step6, "topn", 10)))
        except Exception:
            pass

    top = out_df.sort_values(["_score", "_prob"], ascending=False).head(topn).copy()

    # ---- Output final readable table ----
    result = pd.DataFrame({
        "ts_code": top[ts_col].astype(str),
        "name": top[name_col].astype(str),
        "score": top["_score"].astype(float).round(6),
        "prob": top["_prob"].astype(float).round(6),
        "board": top["_board"],
        "StrengthScore": top["_strength"].astype(float).round(3),
        "ThemeBoost": top["_theme_boost"].astype(float).round(6),
    }).reset_index(drop=True)

    return result


# Backward compatible alias
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 Final TopN ready.")
