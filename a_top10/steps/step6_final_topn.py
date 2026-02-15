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
    - w_prob / w_strength / w_theme: float 权重（默认 1.0/0.0/0.0）
    - min_prob: float，低于该概率直接过滤（默认 0.0 不过滤）
    - st_penalty: float，ST 惩罚因子（默认 0.85）
    - dedup_by_ts_code: bool，是否按 ts_code 去重（默认 True）
    - theme_cap: float，theme 归一化上限（默认 1.3）
    - emotion_factor: float，全局乘子（默认 1.0）
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any, Mapping
import numpy as np
import pandas as pd


# -------------------------
# Input normalize / guard
# -------------------------
def _coerce_df(obj: Any) -> pd.DataFrame:
    """
    统一把输入转换成 DataFrame：
    - DataFrame: 原样返回
    - dict: 尝试从常见键名取 DataFrame；否则遍历 values 找第一个 DataFrame
    - 其他类型：抛 TypeError
    """
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, Mapping):
        for k in ("df", "data", "result", "full", "candidates", "candidate", "pool", "merged"):
            v = obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
        raise TypeError(
            f"Step6 run_step6_final_topn() 收到 dict，但在其常见键/values 中找不到 DataFrame。keys={list(obj.keys())}"
        )
    raise TypeError(
        f"Step6 run_step6_final_topn() 期望 DataFrame 或 dict(内含 DataFrame)，但收到类型：{type(obj)}"
    )


# -------------------------
# Helpers
# -------------------------
def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or len(df) == 0:
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


def _as_str_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.where(~pd.isna(s), pd.NA)
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: pd.NA if isinstance(x, str) and x == "" else x)
    return s


def _is_effectively_empty(s: pd.Series) -> bool:
    if s is None or len(s) == 0:
        return True
    s2 = s.astype("object")
    s2 = s2.map(lambda x: x.strip() if isinstance(x, str) else x)
    s2 = s2.map(lambda x: pd.NA if isinstance(x, str) and x == "" else x)
    return s2.isna().all()


def _ensure_identity_columns(out: pd.DataFrame, ts_col: str, name_col: str) -> pd.DataFrame:
    """
    确保 ts_code / name 至少有“可用值”（列存在但全空白也算缺失）
    - 这里不强制改列名，只确保最终能取到字符串值
    """
    if out is None or len(out) == 0:
        return out

    # 清洗现有
    for c in [ts_col, name_col]:
        if c in out.columns:
            out[c] = _as_str_series(out, c)

    # ts_code fallback
    if (ts_col not in out.columns) or _is_effectively_empty(out[ts_col]):
        for c in ["ts_code", "代码", "code", "symbol", "ticker", "证券代码", "TS_CODE"]:
            if c in out.columns:
                cand = _as_str_series(out, c)
                if not _is_effectively_empty(cand):
                    out[ts_col] = cand
                    break
        if ts_col not in out.columns:
            out[ts_col] = ""

    # name fallback
    if (name_col not in out.columns) or _is_effectively_empty(out[name_col]):
        for c in ["name", "名称", "stock_name", "证券名称", "NAME", "股票"]:
            if c in out.columns:
                cand = _as_str_series(out, c)
                if not _is_effectively_empty(cand):
                    out[name_col] = cand
                    break
        if name_col not in out.columns:
            out[name_col] = ""

    out[ts_col] = _as_str_series(out, ts_col)
    out[name_col] = _as_str_series(out, name_col)
    return out


def _get_setting(s, names: Sequence[str], default):
    """
    支持:
      - s.attr
      - s["key"]  (dict / Mapping)
    """
    if s is None:
        return default

    if isinstance(s, Mapping):
        for k in names:
            if k in s:
                return s.get(k, default)
            lk = str(k).lower()
            if lk in s:
                return s.get(lk, default)
        return default

    for k in names:
        if hasattr(s, k):
            try:
                return getattr(s, k)
            except Exception:
                pass
        lk = str(k).lower()
        if hasattr(s, lk):
            try:
                return getattr(s, lk)
            except Exception:
                pass
    return default


def _clip01(x: pd.Series) -> pd.Series:
    return x.astype("float64").clip(0.0, 1.0)


def top_df_raw_toggle_empty(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.where(~pd.isna(s), "")
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: "" if (x is None) or (isinstance(x, str) and x.lower() == "nan") else x)
    return s.astype("object")


# =========================================================
#                    Step6 Final Scoring
# =========================================================
def run_step6_final_topn(df: Any, s=None) -> Dict[str, pd.DataFrame]:
    df = _coerce_df(df)
    if df is None or len(df) == 0:
        empty = pd.DataFrame()
        return {"topN": empty, "topn": empty, "full": empty}

    out = df.copy()

    # -------------------------------------------------------
    # ① 自动字段识别 + 兜底列
    #    board/theme 这列在你链路里常常其实是 industry，所以这里把 industry 也纳入候选
    # -------------------------------------------------------
    ts_col = _first_existing_col(out, ["ts_code", "TS_CODE", "code", "symbol", "ticker"]) or "ts_code"
    name_col = _first_existing_col(out, ["name", "NAME", "stock_name", "名称", "证券名称"]) or "name"
    board_col = _first_existing_col(out, ["board", "BOARD", "industry", "INDUSTRY", "板块", "行业", "concept", "theme"]) or "board"

    if ts_col not in out.columns:
        out[ts_col] = ""
    if name_col not in out.columns:
        out[name_col] = ""
    if board_col not in out.columns:
        out[board_col] = ""

    # 确保身份列可用（列名不改，只保证值不是全空白）
    out = _ensure_identity_columns(out, ts_col=ts_col, name_col=name_col)

    # -------------------------------------------------------
    # ② 读取核心字段（尽量兼容各步输出）
    # -------------------------------------------------------
    prob_col = _first_existing_col(out, ["Probability", "probability", "prob", "_prob", "proba", "p"])
    str_col = _first_existing_col(out, ["StrengthScore", "strengthscore", "strength", "_strength"])
    thm_col = _first_existing_col(out, ["ThemeBoost", "themeboost", "theme_boost", "theme", "_theme"])

    prob = _to_float_series(out, prob_col, 0.0).clip(0.0, 1.0)
    strength = _to_float_series(out, str_col, 0.0).clip(0.0, 200.0)  # 常见 0~100，留余量
    theme = _to_float_series(out, thm_col, 1.0).clip(0.0, 10.0)      # 常见 0.8~1.3，留余量

    out["_prob"] = prob
    out["_strength"] = strength
    out["_theme"] = theme

    # -------------------------------------------------------
    # ③ ST penalty（更鲁棒识别）
    # -------------------------------------------------------
    st_penalty_default = float(_get_setting(s, ["st_penalty", "ST_PENALTY"], 0.85))
    st_col = _first_existing_col(out, ["is_st", "isST", "st", "st_flag", "ST", "风险ST"])

    if st_col is not None:
        st_series = _to_float_series(out, st_col, 0.0)
        st_flag = (st_series > 0.5).astype("float64")
    else:
        # 退一步：用 name 判定（中文市场常见：ST、*ST）
        name_series = top_df_raw_toggle_empty(out, name_col).astype(str)

        # ✅ 修复点：去掉“捕获组”避免 pandas 的 UserWarning；并加 na=False 更稳
        # 原：r"(^\*?ST)|(\bST\b)"  -> 会产生捕获组警告
        st_flag = name_series.str.contains(r"^\*?ST|\bST\b", regex=True, na=False).astype("float64")

    st_penalty = pd.Series(
        np.where(st_flag.values > 0.5, st_penalty_default, 1.0),
        index=out.index,
        dtype="float64",
    )
    out["_st_flag"] = st_flag
    out["_st_penalty"] = st_penalty

    # -------------------------------------------------------
    # ④ Emotion factor（全局乘子）
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
    # -------------------------------------------------------
    score_mode = str(_get_setting(s, ["score_mode", "SCORE_MODE"], "geo")).lower()

    w_prob = float(_get_setting(s, ["w_prob", "W_PROB"], 1.0))
    w_strength = float(_get_setting(s, ["w_strength", "W_STRENGTH"], 0.0))
    w_theme = float(_get_setting(s, ["w_theme", "W_THEME"], 0.0))

    w_sum = max(1e-9, (w_prob + w_strength + w_theme))
    w_prob, w_strength, w_theme = w_prob / w_sum, w_strength / w_sum, w_theme / w_sum

    prob01 = _clip01(out["_prob"])

    if score_mode == "add":
        base_score = (w_prob * prob01 + w_strength * strength01 + w_theme * theme01)
    elif score_mode == "mul":
        base_score = (prob01 * strength01 * theme01)
    else:
        # geo: exp( w1*log(x1+eps) + ... )，更稳、更抗极端
        eps = 1e-9
        base_score = np.exp(
            (w_prob * np.log(prob01 + eps))
            + (w_strength * np.log(strength01 + eps))
            + (w_theme * np.log(theme01 + eps))
        )
        base_score = pd.Series(base_score, index=out.index, dtype="float64")

    final_score = base_score * out["_st_penalty"] * emotion_factor

    out["_base_score"] = pd.Series(base_score, index=out.index, dtype="float64")
    out["_score"] = pd.Series(final_score, index=out.index, dtype="float64")

    # -------------------------------------------------------
    # ⑧ 可选去重（同一 ts_code 只保留最高分）
    # -------------------------------------------------------
    dedup = bool(_get_setting(s, ["dedup_by_ts_code", "DEDUP_BY_TS_CODE"], True))
    if dedup and ts_col in out.columns:
        out.sort_values(["_score", "_prob", "_strength", "_theme"], ascending=False, inplace=True)
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

        # 最终分 + 分项
        "score": top_df_raw["_score"].round(6),
        "prob": top_df_raw["_prob"].round(6),
        "StrengthScore": top_df_raw["_strength"].round(3),
        "ThemeBoost": top_df_raw["_theme"].round(3),

        # 可解释字段
        "st_flag": top_df_raw["_st_flag"].round(0).astype(int),
        "st_penalty": top_df_raw["_st_penalty"].round(3),
        "base_score": top_df_raw["_base_score"].round(6),
    })

    return {
        "topN": top_df,
        "topn": top_df,      # 向后兼容
        "full": full_sorted,
    }


def run(df: Any, s=None) -> Dict[str, pd.DataFrame]:
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 FinalTopN v3.2 ready.")
