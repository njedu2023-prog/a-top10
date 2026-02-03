#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Final Selector (TopN + Full Ranking)
-------------------------------------------
综合 Step1~5 的结果，形成最终评分：

    FinalScore = Probability × StrengthScore × ThemeBoost × ST_penalty × EmotionFactor(可选)

新增能力：
1) 输出 TopN（用于实盘/预测）
2) 输出 full 排序（当前候选全集，不是全市场）
3) 所有字段自动兜底，保证永不报错
4) 可解释：输出所有中间字段（_prob、_strength、_theme_boost…）
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
    """将列转为 float，支持 NaN / inf 自动清洗"""
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
    FinalScore = prob × strength × theme × st_penalty × emotion_factor(optional)

    返回:
        {
          "topN": DataFrame(TopN),
          "full": DataFrame(全候选全集排序)
        }
    """
    if df is None or len(df) == 0:
        return {
            "topN": pd.DataFrame(),
            "full": pd.DataFrame()
        }

    out = df.copy()

    # -------------------------------------------------------
    # ① 自动识别必要字段
    # -------------------------------------------------------
    ts_col = _first_existing_col(out, ["ts_code", "TS_CODE", "code"]) or "ts_code"
    name_col = _first_existing_col(out, ["name", "NAME"]) or "name"
    board_col = _first_existing_col(out, ["board", "BOARD", "板块"]) or "board"

    if ts_col not in out: out[ts_col] = ""
    if name_col not in out: out[name_col] = ""
    if board_col not in out: out[board_col] = ""

    # -------------------------------------------------------
    # ② 核心数值字段
    # -------------------------------------------------------
    prob = _to_float_series(out, _first_existing_col(out, ["Probability", "prob", "_prob"]), 0.0)
    strength = _to_float_series(out, _first_existing_col(out, ["StrengthScore", "strength"]), 0.0)
    theme = _to_float_series(out, _first_existing_col(out, ["ThemeBoost", "theme_boost"]), 1.0)

    # 合理 Clamp
    theme = theme.clip(0.0, 10.0)

    # -------------------------------------------------------
    # ③ ST 惩罚（轻惩罚，不剔除）
    # -------------------------------------------------------
    st_col = _first_existing_col(out, ["is_st", "st", "ST"])
    if st_col:
        is_st = _to_float_series(out, st_col, 0.0)
        st_penalty = np.where(is_st > 0.5, 0.85, 1.0)
    else:
        st_penalty = 1.0

    # -------------------------------------------------------
    # ④ Emotion 因子（来自 Step1，可选）
    # -------------------------------------------------------
    emotion_factor = 1.0
    if s is not None and hasattr(s, "emotion_factor"):
        try:
            emotion_factor = float(s.emotion_factor)
        except:
            emotion_factor = 1.0

    # -------------------------------------------------------
    # ⑤ Final Score
    # -------------------------------------------------------
    final_score = prob * strength * theme
    if isinstance(st_penalty, np.ndarray):
        final_score *= st_penalty
    final_score *= emotion_factor

    # 写入中间字段，便于回测查看
    out["_prob"] = prob
    out["_strength"] = strength
    out["_theme"] = theme
    out["_st_penalty"] = st_penalty
    out["_emotion_factor"] = emotion_factor
    out["_score"] = final_score

    # -------------------------------------------------------
    # ⑥ 排序（全集 full 排序）
    # -------------------------------------------------------
    full_sorted = out.sort_values(["_score", "_prob"], ascending=False).copy()
    full_sorted = full_sorted.reset_index(drop=True)

    # -------------------------------------------------------
    # ⑦ 选取 TopN
    # -------------------------------------------------------
    topN = 10
    if s is not None:
        if hasattr(s, "topn"):
            try:
                topN = max(1, int(s.topn))
            except:
                pass
        elif hasattr(s, "step6"):
            try:
                topN = max(1, int(getattr(s.step6, "topn", 10)))
            except:
                pass

    top_df = full_sorted.head(topN).copy()

    # 输出精简可读格式
    top_df = pd.DataFrame({
        "ts_code": top_df[ts_col].astype(str),
        "name": top_df[name_col].astype(str),
        "score": top_df["_score"].round(6),
        "prob": top_df["_prob"].round(6),
        "board": top_df[board_col].astype(str),
        "StrengthScore": top_df["_strength"].round(3),
        "ThemeBoost": top_df["_theme"].round(3),
    })

    return {
        "topN": top_df,
        "full": full_sorted  # 保留全部字段供回测/调参/诊断
    }


# 兼容旧主程序
def run(df: pd.DataFrame, s=None):
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 FinalTopN v3.0 ready.")
