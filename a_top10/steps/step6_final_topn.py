#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Final Selector (TopN + Full Ranking) — Engine 3.0 完整升级版
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
    lower_map = {str(c).lower(): c for c in df.columns}
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


# =========================================================
#                Step6 Final Scoring 3.0
# =========================================================
def run_step6_final_topn(df: pd.DataFrame, s=None):
    """
    返回:
        {
           "topn": DataFrame(精简 TopN),
           "full": DataFrame(完整排序全集)
        }
    """
    if df is None or len(df) == 0:
        return {"topn": pd.DataFrame(), "full": pd.DataFrame()}

    out = df.copy()

    # -------------------------------------------------------
    # ① 自动字段识别
    # -------------------------------------------------------
    ts_col = _first_existing_col(out, ["ts_code", "TS_CODE", "code"]) or "ts_code"
    name_col = _first_existing_col(out, ["name", "NAME"]) or "name"
    board_col = _first_existing_col(out, ["board", "BOARD", "板块"]) or "board"

    for col, default in [(ts_col, ""), (name_col, ""), (board_col, "")]:
        if col not in out:
            out[col] = default

    # -------------------------------------------------------
    # ② 核心字段
    # -------------------------------------------------------
    prob = _to_float_series(out, _first_existing_col(out, ["Probability", "prob", "_prob"]), 0.0)
    strength = _to_float_series(out, _first_existing_col(out, ["StrengthScore", "strength"]), 0.0)
    theme = _to_float_series(out, _first_existing_col(out, ["ThemeBoost", "theme_boost"]), 1.0)
    theme = theme.clip(0.0, 10.0)

    # -------------------------------------------------------
    # ③ ST penalty
    # -------------------------------------------------------
    st_col = _first_existing_col(out, ["is_st", "st", "ST"])
    if st_col:
        st_series = _to_float_series(out, st_col, 0.0)
        st_penalty = pd.Series(
            np.where(st_series > 0.5, 0.85, 1.0),
            index=out.index,
            dtype="float64",
        )
    else:
        st_penalty = pd.Series([1.0] * len(out), index=out.index, dtype="float64")

    # -------------------------------------------------------
    # ④ Emotion factor
    # -------------------------------------------------------
    emotion_factor = 1.0
    if s is not None and hasattr(s, "emotion_factor"):
        try:
            emotion_factor = float(s.emotion_factor)
        except:
            pass
    emotion_factor = float(emotion_factor)

    # -------------------------------------------------------
    # ⑤ FinalScore
    # -------------------------------------------------------
    final_score = prob * strength * theme * st_penalty * emotion_factor
    out["_prob"] = prob
    out["_strength"] = strength
    out["_theme"] = theme
    out["_st_penalty"] = st_penalty
    out["_emotion_factor"] = emotion_factor
    out["_score"] = final_score

    # -------------------------------------------------------
    # ⑥ full 排序
    # -------------------------------------------------------
    full_sorted = out.sort_values(["_score", "_prob"], ascending=False).copy()
    full_sorted.reset_index(drop=True, inplace=True)
    full_sorted["rank"] = full_sorted.index + 1  # 排名

    # -------------------------------------------------------
    # ⑦ TopN
    # -------------------------------------------------------
    topN = 10
    if s is not None:
        for k in ["topn", "TOPN"]:
            if hasattr(s, k):
                try:
                    topN = max(1, int(getattr(s, k)))
                except:
                    pass

    top_df_raw = full_sorted.head(topN).copy()

    # 输出精简格式（用于 UI / markdown）
    top_df = pd.DataFrame({
        "rank": top_df_raw["rank"],
        "ts_code": top_df_raw[ts_col].astype(str),
        "name": top_df_raw[name_col].astype(str),
        "score": top_df_raw["_score"].round(6),
        "prob": top_df_raw["_prob"].round(6),
        "StrengthScore": top_df_raw["_strength"].round(3),
        "ThemeBoost": top_df_raw["_theme"].round(3),
        "board": top_df_raw[board_col].astype(str),
    })

    return {
        "topn": top_df,       # 精简版（10 条）
        "full": full_sorted,  # 完整排序全集
    }


def run(df: pd.DataFrame, s=None):
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 FinalTopN v3.1 ready.")
