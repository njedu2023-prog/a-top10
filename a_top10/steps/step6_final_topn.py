# -*- coding: utf-8 -*-
"""
Step6: final topN selection.

Design goals:
- No hard "gate" that can make TopN empty.
- Prefer upstream real fields: Probability/StrengthScore/ThemeBoost.
- Robust to missing columns (fallbacks).
- Produce explainable outputs: final_score/prob/strength/theme_boost/board/risk_penalty/risk_tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name that exists in df (case-insensitive match)."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    """Safely convert df[col] to float series."""
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


def _clip(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def _robust_minmax(s: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """
    Robust min-max scaling into [0,1] using quantiles to reduce outlier effects.
    If constant / invalid, return 0.5.
    """
    if s is None or len(s) == 0:
        return pd.Series([], dtype="float64")

    x = pd.to_numeric(s, errors="coerce").astype("float64")
    x = x.replace([np.inf, -np.inf], np.nan)

    if x.notna().sum() == 0:
        return pd.Series([0.5] * len(x), index=x.index, dtype="float64")

    lo = float(x.quantile(q_low))
    hi = float(x.quantile(q_high))

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return pd.Series([0.5] * len(x), index=x.index, dtype="float64")

    y = (x - lo) / (hi - lo)
    y = y.clip(0.0, 1.0).fillna(0.5)
    return y


def _normalize_prob(prob: pd.Series) -> pd.Series:
    """
    Normalize probability into [0,1] with robust handling.
    Accepts:
      - already in [0,1]
      - or in [0,100]
      - or messy values -> clip
    """
    p = pd.to_numeric(prob, errors="coerce").astype("float64").fillna(0.0)
    p = p.replace([np.inf, -np.inf], 0.0)

    # Heuristic: if most values > 1.5, treat as percent
    if (p > 1.5).mean() > 0.5:
        p = p / 100.0

    return p.clip(0.0, 1.0)


def _ensure_board(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort board field. If no board col, infer from ts_code prefix:
      60xxxx -> SH
      00/30xxxx -> SZ
      688xxx -> STAR
      83/87/88/43.. -> BJ (rough)
    """
    board_col = _first_existing_col(df, ["board", "market", "exchange"])
    if board_col:
        return df[board_col].astype(str)

    ts_col = _first_existing_col(df, ["ts_code", "symbol", "code"])
    if not ts_col:
        return pd.Series(["UNK"] * len(df), index=df.index)

    codes = df[ts_col].astype(str).fillna("")
    out = []
    for c in codes:
        cc = c.split(".")[0]  # handle 000001.SZ
        if cc.startswith("60"):
            out.append("SH")
        elif cc.startswith("688"):
            out.append("STAR")
        elif cc.startswith("00") or cc.startswith("30"):
            out.append("SZ")
        elif cc.startswith(("83", "87", "88", "43")):
            out.append("BJ")
        else:
            out.append("UNK")
    return pd.Series(out, index=df.index)


def _safe_str_series(df: pd.DataFrame, col: Optional[str], default: str = "") -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[col].astype(str).fillna(default)


# -----------------------------
# Settings bridge (optional)
# -----------------------------

@dataclass
class Step6Settings:
    topn: int = 10


def _get_step6_settings(s) -> Step6Settings:
    """
    Try to read settings from pipeline context object `s`.
    Compatible patterns:
      - s.io.topn
      - s.io.step6.topn
      - s.config.get("topn")
    """
    topn = 10

    try:
        v = getattr(getattr(s, "io", None), "topn", None)
        if v is not None:
            topn = int(v)
    except Exception:
        pass

    try:
        v = getattr(getattr(getattr(s, "io", None), "step6", None), "topn", None)
        if v is not None:
            topn = int(v)
    except Exception:
        pass

    try:
        cfg = getattr(s, "config", None)
        if isinstance(cfg, dict) and "topn" in cfg:
            topn = int(cfg.get("topn") or topn)
    except Exception:
        pass

    if topn <= 0:
        topn = 10
    return Step6Settings(topn=topn)


# -----------------------------
# Risk penalty helpers (soft, no hard gate)
# -----------------------------

def _penalty_st(df: pd.DataFrame) -> np.ndarray:
    st_col = _first_existing_col(df, ["is_st", "st", "ST", "risk_is_st"])
    if not st_col:
        return np.ones(len(df), dtype="float64")
    is_st = _to_float_series(df, st_col, 0.0).values
    return np.where(is_st > 0.5, 0.85, 1.0).astype("float64")


def _penalty_dt_net_sell(df: pd.DataFrame) -> np.ndarray:
    """
    龙虎榜净卖出惩罚：
    支持字段：
      - dt_net_sell_amount / 龙虎榜净卖出额 / net_sell
      - dt_net_sell_ratio / 净卖出占比
    规则：越大越惩罚，但不硬剔除。
    """
    amt_col = _first_existing_col(df, ["dt_net_sell_amount", "龙虎榜净卖出额", "net_sell_amount", "net_sell"])
    ratio_col = _first_existing_col(df, ["dt_net_sell_ratio", "净卖出占比", "net_sell_ratio"])

    if (not amt_col) and (not ratio_col):
        return np.ones(len(df), dtype="float64")

    pen = np.ones(len(df), dtype="float64")

    if ratio_col:
        r = _to_float_series(df, ratio_col, 0.0).values
        # ratio: 0~1, >0.2明显卖压
        pen = pen * np.where(r >= 0.35, 0.85, np.where(r >= 0.20, 0.90, np.where(r >= 0.10, 0.95, 1.0)))

    if amt_col:
        a = _to_float_series(df, amt_col, 0.0).values
        # 单位不确定：用量级做温和惩罚（只要 >0 就说明净卖出）
        pen = pen * np.where(a >= 2e8, 0.85, np.where(a >= 5e7, 0.90, np.where(a > 0, 0.95, 1.0)))

    return pen.astype("float64")


def _penalty_high_accel(df: pd.DataFrame) -> np.ndarray:
    """
    高位加速风险惩罚：
    支持字段：
      - high_accel_risk / 高位加速风险 / accel_risk
      - 或 high_risk_score (0~1)
    """
    flag_col = _first_existing_col(df, ["high_accel_risk", "高位加速风险", "accel_risk", "risk_high_accel"])
    score_col = _first_existing_col(df, ["high_risk_score", "risk_score_high", "accel_risk_score"])

    if (not flag_col) and (not score_col):
        return np.ones(len(df), dtype="float64")

    if score_col:
        sc = _to_float_series(df, score_col, 0.0).values
        sc = np.clip(sc, 0.0, 1.0)
        return (1.0 - 0.2 * sc).astype("float64")  # 0.8~1.0

    fl = _to_float_series(df, flag_col, 0.0).values
    return np.where(fl > 0.5, 0.85, 1.0).astype("float64")


def _penalty_nuke(df: pd.DataFrame) -> np.ndarray:
    """
    次日核按钮风险惩罚：
    支持字段：
      - nuke_risk / 核按钮风险 / nextday_nuke_risk
      - 或 nuke_risk_score (0~1)
    """
    flag_col = _first_existing_col(df, ["nuke_risk", "核按钮风险", "nextday_nuke_risk", "risk_nuke"])
    score_col = _first_existing_col(df, ["nuke_risk_score", "risk_score_nuke"])

    if (not flag_col) and (not score_col):
        return np.ones(len(df), dtype="float64")

    if score_col:
        sc = _to_float_series(df, score_col, 0.0).values
        sc = np.clip(sc, 0.0, 1.0)
        return (1.0 - 0.25 * sc).astype("float64")  # 0.75~1.0

    fl = _to_float_series(df, flag_col, 0.0).values
    return np.where(fl > 0.5, 0.80, 1.0).astype("float64")


def _build_risk_tags(df: pd.DataFrame) -> pd.Series:
    """
    输出字符串标签，便于解释
    """
    tags = [[] for _ in range(len(df))]

    st_col = _first_existing_col(df, ["is_st", "st", "ST", "risk_is_st"])
    if st_col:
        v = _to_float_series(df, st_col, 0.0).values
        for i, x in enumerate(v):
            if x > 0.5:
                tags[i].append("ST")

    amt_col = _first_existing_col(df, ["dt_net_sell_amount", "龙虎榜净卖出额", "net_sell_amount", "net_sell"])
    ratio_col = _first_existing_col(df, ["dt_net_sell_ratio", "净卖出占比", "net_sell_ratio"])
    if amt_col or ratio_col:
        a = _to_float_series(df, amt_col, 0.0).values if amt_col else None
        r = _to_float_series(df, ratio_col, 0.0).values if ratio_col else None
        for i in range(len(df)):
            flag = False
            if r is not None and r[i] >= 0.20:
                flag = True
            if a is not None and a[i] > 0:
                flag = True
            if flag:
                tags[i].append("DT_NET_SELL")

    ha_col = _first_existing_col(df, ["high_accel_risk", "高位加速风险", "accel_risk", "risk_high_accel"])
    ha_sc = _first_existing_col(df, ["high_risk_score", "risk_score_high", "accel_risk_score"])
    if ha_col or ha_sc:
        v = _to_float_series(df, ha_sc or ha_col, 0.0).values
        for i, x in enumerate(v):
            if x > 0.5:
                tags[i].append("HIGH_ACCEL")

    nk_col = _first_existing_col(df, ["nuke_risk", "核按钮风险", "nextday_nuke_risk", "risk_nuke"])
    nk_sc = _first_existing_col(df, ["nuke_risk_score", "risk_score_nuke"])
    if nk_col or nk_sc:
        v = _to_float_series(df, nk_sc or nk_col, 0.0).values
        for i, x in enumerate(v):
            if x > 0.5:
                tags[i].append("NUKE_RISK")

    out = [",".join(x) if x else "" for x in tags]
    return pd.Series(out, index=df.index, dtype="object")


# -----------------------------
# Main step
# -----------------------------

def run_step6_final_topn(df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    Input: df includes at least ts_code/name + Probability/StrengthScore/ThemeBoost (preferred).
    Output: topN DataFrame with explainable columns.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "ts_code", "name", "final_score", "prob", "board",
                "StrengthScore", "ThemeBoost",
                "risk_penalty", "risk_tags"
            ]
        )

    # Required-ish columns
    ts_col = _first_existing_col(df, ["ts_code", "symbol", "code"])
    name_col = _first_existing_col(df, ["name", "stock_name", "sec_name", "证券简称"])
    if ts_col is None:
        df = df.copy()
        df["ts_code"] = df.index.astype(str)
        ts_col = "ts_code"
    if name_col is None:
        df = df.copy()
        df["name"] = df[ts_col].astype(str)
        name_col = "name"

    # -------------------------
    # 1) Probability (primary)
    # -------------------------
    prob_col = _first_existing_col(df, ["Probability", "prob", "probability", "p", "_prob", "PredProb", "预测概率"])
    prob = _normalize_prob(_to_float_series(df, prob_col, 0.5)) if prob_col else pd.Series(
        [0.5] * len(df), index=df.index, dtype="float64"
    )

    # -------------------------
    # 2) StrengthScore (prefer step3)
    # -------------------------
    strength_col = _first_existing_col(df, ["StrengthScore", "_strength", "strength", "score_strength"])
    if strength_col:
        strength = _to_float_series(df, strength_col, 50.0)
        # 若上游是 0~1 可能性，则转 0~100
        if (strength <= 1.5).mean() > 0.8:
            strength = strength * 100.0
        strength = strength.clip(0.0, 100.0)
    else:
        # fallback: old proxy from pct/vol/amt/turnover (last resort)
        pct_col = _first_existing_col(df, ["pct_chg", "pct", "change_pct", "涨跌幅"])
        volr_col = _first_existing_col(df, ["vol_ratio", "volume_ratio", "量比", "volr"])
        amt_col = _first_existing_col(df, ["amount", "turnover", "amt", "成交额"])
        tr_col = _first_existing_col(df, ["turnover_rate", "turn_rate", "换手率"])

        x_pct = _robust_minmax(_to_float_series(df, pct_col, 0.0)) if pct_col else pd.Series([0.5]*len(df), index=df.index)
        x_vol = _robust_minmax(_to_float_series(df, volr_col, 1.0)) if volr_col else pd.Series([0.5]*len(df), index=df.index)
        x_amt = _robust_minmax(_to_float_series(df, amt_col, 0.0)) if amt_col else pd.Series([0.5]*len(df), index=df.index)
        x_tr  = _robust_minmax(_to_float_series(df, tr_col, 0.0))  if tr_col  else pd.Series([0.5]*len(df), index=df.index)

        strength01 = 0.40 * x_pct + 0.25 * x_vol + 0.25 * x_amt + 0.10 * x_tr
        strength01 = _clip(strength01, 0.0, 1.0)
        strength = strength01 * 100.0

    # -------------------------
    # 3) ThemeBoost (prefer step4)
    # -------------------------
    theme_col = _first_existing_col(df, ["ThemeBoost", "theme_boost", "_theme_boost", "themeScore"])
    if theme_col:
        theme_boost = _to_float_series(df, theme_col, 1.0)
        # keep sane range
        theme_boost = theme_boost.replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.0, 10.0)
    else:
        theme_boost = pd.Series([1.0] * len(df), index=df.index, dtype="float64")

    # board
    board = _ensure_board(df)

    # -------------------------
    # 4) Risk penalties (soft, multiplicative)
    # -------------------------
    p_st = _penalty_st(df)
    p_dt = _penalty_dt_net_sell(df)
    p_ha = _penalty_high_accel(df)
    p_nk = _penalty_nuke(df)

    risk_penalty = (p_st * p_dt * p_ha * p_nk).astype("float64")
    # 不让惩罚过度失真
    risk_penalty = np.clip(risk_penalty, 0.50, 1.00)

    risk_tags = _build_risk_tags(df)

    # -------------------------
    # 5) Final score
    # -------------------------
    # 核心公式：FinalScore = Probability * StrengthScore * ThemeBoost * RiskPenalty
    final_score = (prob.values * strength.values * theme_boost.values * risk_penalty).astype("float64")

    # attach debug columns
    out_df = df.copy()
    out_df["_prob"] = prob
    out_df["_strength"] = strength
    out_df["_theme_boost"] = theme_boost
    out_df["_risk_penalty"] = risk_penalty
    out_df["_final_score"] = final_score
    out_df["_board"] = board
    out_df["_risk_tags"] = risk_tags

    # select topN
    settings = _get_step6_settings(s)
    topn = int(settings.topn or 10)
    topn = max(1, topn)

    top = out_df.sort_values(["_final_score", "_prob"], ascending=False).head(topn).copy()

    result = pd.DataFrame(
        {
            "ts_code": _safe_str_series(top, ts_col),
            "name": _safe_str_series(top, name_col),
            "final_score": pd.to_numeric(top["_final_score"], errors="coerce").fillna(0.0).astype(float).round(6),
            "prob": pd.to_numeric(top["_prob"], errors="coerce").fillna(0.0).astype(float).round(6),
            "board": top["_board"].astype(str),
            "StrengthScore": pd.to_numeric(top["_strength"], errors="coerce").fillna(0.0).astype(float).round(3),
            "ThemeBoost": pd.to_numeric(top["_theme_boost"], errors="coerce").fillna(1.0).astype(float).round(6),
            "risk_penalty": pd.to_numeric(top["_risk_penalty"], errors="coerce").fillna(1.0).astype(float).round(6),
            "risk_tags": top["_risk_tags"].astype(str),
        }
    ).reset_index(drop=True)

    return result


# Backward-compatible alias if your pipeline imports a different function name
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step6_final_topn(df, s=s)
