#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Final Selector (TopN + Full Ranking) — Engine 3.2 (稳定增强版)

输出:
    {
        "topN": DataFrame(精简 TopN)   # 推荐主 key
        "topn": DataFrame(精简 TopN)   # 向后兼容
        "full": DataFrame(完整排序全集)
    }

支持 settings(s) 可选参数（均可缺省）:
    - topn / TOPN: int，默认 10
    - score_mode: "geo" | "mul" | "add"   （默认 "geo"）
    - w_prob / w_strength / w_theme: float 权重（默认 0.85/0.12/0.03）
    - min_prob: float，低于该概率直接过滤（默认 0.0 不过滤）
    - st_penalty: float，ST 惩罚因子（默认 0.85）
    - dedup_by_ts_code: bool，是否按 ts_code 去重（默认 True）
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any, Tuple
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


def _get_setting(s, names: Sequence[str], default):
    if s is None:
        return default
    for k in names:
        if hasattr(s, k):
            try:
                v = getattr(s, k)
                return v
            except Exception:
                pass
    return default


def _clip01(x: pd.Series) -> pd.Series:
    return x.astype("float64").clip(0.0, 1.0)


def _safe_pow(x: pd.Series, p: float) -> pd.Series:
    # 避免 0**负数 或 nan 扩散
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.clip(0.0, 1.0)
    p = float(p)
    return x.pow(p)


# =========================================================
#                    Step6 Final Scoring
# =========================================================
def run_step6_final_topn(df: pd.DataFrame, s=None) -> Dict[str, pd.DataFrame]:
    if df is None or len(df) == 0:
        empty = pd.DataFrame()
        return {"topN": empty, "topn": empty, "full": empty}

    out = df.copy()

    # -------------------------------------------------------
    # ① 自动字段识别 + 兜底列
    # -------------------------------------------------------
    ts_col = _first_existing_col(out, ["ts_code", "TS_CODE", "code", "symbol"]) or "ts_code"
    name_col = _first_existing_col(out, ["name", "NAME", "stock_name", "名称"]) or "name"
    board_col = _first_existing_col(out, ["board", "BOARD", "板块", "concept", "theme"]) or "board"

    if ts_col not in out.columns:
        out[ts_col] = ""
    if name_col not in out.columns:
        out[name_col] = ""
    if board_col not in out.columns:
        out[board_col] = ""

    # -------------------------------------------------------
    # ② 读取核心字段（尽量兼容各步输出）
    # -------------------------------------------------------
    prob_col = _first_existing_col(out, ["Probability", "prob", "_prob", "proba"])
    str_col = _first_existing_col(out, ["StrengthScore", "strength", "_strength"])
    thm_col = _first_existing_col(out, ["ThemeBoost", "theme_boost", "theme", "_theme"])

    prob = _to_float_series(out, prob_col, 0.0)
    strength = _to_float_series(out, str_col, 0.0)
    theme = _to_float_series(out, thm_col, 1.0)

    # 合理裁剪
    prob = prob.clip(0.0, 1.0)
    strength = strength.clip(0.0, 200.0)      # StrengthScore 常见 0~100，给点余量
    theme = theme.clip(0.0, 10.0)             # ThemeBoost 常见 0.8~1.3，给点余量

    out["_prob"] = prob
    out["_strength"] = strength
    out["_theme"] = theme

    # -------------------------------------------------------
    # ③ ST penalty（更鲁棒识别）
    # -------------------------------------------------------
    st_penalty_default = float(_get_setting(s, ["st_penalty", "ST_PENALTY"], 0.85))
    st_col = _first_existing_col(out, ["is_st", "st", "ST", "isST", "st_flag", "风险ST"])

    if st_col is not None:
        st_series = _to_float_series(out, st_col, 0.0)
        st_flag = (st_series > 0.5).astype("float64")
    else:
        # 退一步：用 name 里包含 ST 判定
        name_series = out[name_col].astype(str)
        st_flag = name_series.str.contains(r"\bST\b|^\*ST|ST", regex=True).astype("float64")

    st_penalty = pd.Series(
        np.where(st_flag.values > 0.5, st_penalty_default, 1.0),
        index=out.index,
        dtype="float64",
    )
    out["_st_flag"] = st_flag
    out["_st_penalty"] = st_penalty

    # -------------------------------------------------------
    # ④ Emotion factor（全局乘子，兼容你原逻辑）
    # -------------------------------------------------------
    emotion_factor = float(_get_setting(s, ["emotion_factor", "EMOTION_FACTOR"], 1.0))
    out["_emotion_factor"] = emotion_factor

    # -------------------------------------------------------
    # ⑤ 可选过滤（例如过滤极低概率）
    # -------------------------------------------------------
    min_prob = float(_get_setting(s, ["min_prob", "MIN_PROB"], 0.0))
    if min_prob > 0:
        out = out[out["_prob"] >= min_prob].copy()
        if out.empty:
            empty = pd.DataFrame()
            return {"topN": empty, "topn": empty, "full": empty}

    # -------------------------------------------------------
    # ⑥ 归一化（让不同尺度可比）
    #   prob: 已是 0~1
    #   strength: 0~100 -> 0~1（超过 100 也会被 clip 到 1）
    #   theme: 以 1.3 视为接近上限（可调整）
    # -------------------------------------------------------
    strength01 = _clip01(out["_strength"] / 100.0)
    theme_cap = float(_get_setting(s, ["theme_cap", "THEME_CAP"], 1.3))
    if theme_cap <= 0:
        theme_cap = 1.3
    theme01 = _clip01(out["_theme"] / theme_cap)

    out["_strength01"] = strength01
    out["_theme01"] = theme01

    # -------------------------------------------------------
    # ⑦ FinalScore（支持三种模式）
    #   geo(默认): 加权几何均值，稳定、抗极值
    #   mul: 你原来的乘法（但仍使用归一化后的分项）
    #   add: 加权加法
    # -------------------------------------------------------
    score_mode = str(_get_setting(s, ["score_mode", "SCORE_MODE"], "geo")).lower()

    w_prob = float(_get_setting(s, ["w_prob", "W_PROB"], 0.85))
    w_strength = float(_get_setting(s, ["w_strength", "W_STRENGTH"], 0.12))
    w_theme = float(_get_setting(s, ["w_theme", "W_THEME"], 0.03))
    # 归一化权重（避免用户乱填）
    w_sum = max(1e-9, (w_prob + w_strength + w_theme))
    w_prob, w_strength, w_theme = w_prob / w_sum, w_strength / w_sum, w_theme / w_sum

    prob01 = _clip01(out["_prob"])

    if score_mode == "add":
        base_score = (w_prob * prob01 + w_strength * strength01 + w_theme * theme01)
    elif score_mode == "mul":
        base_score = (prob01 * strength01 * theme01)
    else:
        # geo: exp( w1*log(x1+eps) + ... ) 更稳
        eps = 1e-9
        base_score = np.exp(
            (w_prob * np.log(prob01 + eps))
            + (w_strength * np.log(strength01 + eps))
            + (w_theme * np.log(theme01 + eps))
        )
        base_score = pd.Series(base_score, index=out.index, dtype="float64")

    final_score = base_score * out["_st_penalty"] * emotion_factor

    out["_base_score"] = base_score
    out["_score"] = final_score

    # -------------------------------------------------------
    # ⑧ 可选去重（同一 ts_code 只保留最高分）
    # -------------------------------------------------------
    dedup = bool(_get_setting(s, ["dedup_by_ts_code", "DEDUP_BY_TS_CODE"], True))
    if dedup and ts_col in out.columns:
        out.sort_values(["_score", "_prob", "_strength"], ascending=False, inplace=True)
        out = out.drop_duplicates(subset=[ts_col], keep="first").copy()

    # -------------------------------------------------------
    # ⑨ full 排序（tie-break 更稳定）
    # -------------------------------------------------------
    full_sorted = out.sort_values(
        ["_score", "_prob", "_strength", "_theme"],
        ascending=False
    ).copy()
    full_sorted.reset_index(drop=True, inplace=True)
    full_sorted["rank"] = full_sorted.index + 1

    # -------------------------------------------------------
    # ⑩ TopN
    # -------------------------------------------------------
    topN = int(_get_setting(s, ["topn", "TOPN"], 10) or 10)
    topN = max(1, topN)

    top_df_raw = full_sorted.head(topN).copy()

    top_df = pd.DataFrame({
        "rank": top_df_raw["rank"].astype(int),
        "ts_code": top_df_raw[ts_col].astype(str),
        "name": top_df_raw_toggle_empty(top_df_raw, name_col),
        "board": top_df_raw_toggle_empty(top_df_raw, board_col),

        # 最终分 + 分项（保留你关心的字段）
        "score": top_df_raw["_score"].round(6),
        "prob": top_df_raw["_prob"].round(6),
        "StrengthScore": top_df_raw["_strength"].round(3),
        "ThemeBoost": top_df_raw["_theme"].round(3),

        # 额外可解释字段
        "st_flag": top_df_raw["_st_flag"].round(0).astype(int),
        "st_penalty": top_df_raw["_st_penalty"].round(3),
        "base_score": top_df_raw["_base_score"].round(6),
    })

    # 兼容两套 key：topN & topn
    return {
        "topN": top_df,
        "topn": top_df,
        "full": full_sorted,
    }


def top_df_raw_toggle_empty(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    s = df[col].astype(str)
    return s.replace("nan", "").fillna("")


def run(df: pd.DataFrame, s=None) -> Dict[str, pd.DataFrame]:
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 FinalTopN v3.2 ready.")
