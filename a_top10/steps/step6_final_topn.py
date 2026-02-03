# -*- coding: utf-8 -*-
"""
Step6: final topN selection.

Design goals:
- No hard "gate" that can make Top10 empty.
- Robust to missing columns (fallbacks).
- Produce explainable outputs: score/prob/strength/theme_boost/board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import math
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


def _to_float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Safely convert df[col] to float series."""
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.fillna(default)
    return s


def _clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


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

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series([0.5] * len(x), index=x.index, dtype="float64")

    y = (x - lo) / (hi - lo)
    y = y.clip(0.0, 1.0)
    y = y.fillna(0.5)
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

    p = p.clip(0.0, 1.0)
    return p


def _ensure_board(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort board field. If no board col, infer from ts_code prefix:
      60xxxx -> SH
      00/30xxxx -> SZ
      68xxxx -> STAR
      83/87/88/43... -> BJ (rough)
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
        if cc.startswith("60") or cc.startswith("688"):
            out.append("SH" if cc.startswith("60") else "STAR")
        elif cc.startswith("00") or cc.startswith("30"):
            out.append("SZ")
        elif cc.startswith(("83", "87", "88", "43")):
            out.append("BJ")
        else:
            out.append("UNK")
    return pd.Series(out, index=df.index)


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

    # s.io.topn
    try:
        v = getattr(getattr(s, "io", None), "topn", None)
        if v is not None:
            topn = int(v)
    except Exception:
        pass

    # s.io.step6.topn
    try:
        v = getattr(getattr(getattr(s, "io", None), "step6", None), "topn", None)
        if v is not None:
            topn = int(v)
    except Exception:
        pass

    # s.config dict-like
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
# Main step
# -----------------------------

def run_step6_final_topn(df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    Input: df includes at least ts_code/name + prob columns (preferred).
    Output: topN DataFrame with explainable columns.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=["ts_code", "name", "score", "prob", "board", "StrengthScore", "ThemeBoost"]
        )

    # Required-ish columns
    ts_col = _first_existing_col(df, ["ts_code", "symbol", "code"])
    name_col = _first_existing_col(df, ["name", "stock_name", "sec_name", "证券简称"])
    if ts_col is None:
        # last-resort: create an index-based code
        df = df.copy()
        df["ts_code"] = df.index.astype(str)
        ts_col = "ts_code"
    if name_col is None:
        df = df.copy()
        df["name"] = df[ts_col].astype(str)
        name_col = "name"

    # prob (primary)
    prob_col = _first_existing_col(df, ["prob", "probability", "p", "_prob", "PredProb", "预测概率"])
    if prob_col:
        prob = _normalize_prob(_to_float_series(df, prob_col, 0.0))
    else:
        # fallback: neutral probability
        prob = pd.Series([0.5] * len(df), index=df.index, dtype="float64")

    # strength features (secondary)
    pct_col = _first_existing_col(df, ["pct_chg", "pct", "change_pct", "涨跌幅"])
    volr_col = _first_existing_col(df, ["vol_ratio", "volume_ratio", "量比", "volr"])
    amt_col = _first_existing_col(df, ["amount", "turnover", "amt", "成交额"])
    tr_col  = _first_existing_col(df, ["turnover_rate", "turn_rate", "换手率"])

    x_pct = _robust_minmax(_to_float_series(df, pct_col, 0.0)) if pct_col else pd.Series([0.5]*len(df), index=df.index)
    x_vol = _robust_minmax(_to_float_series(df, volr_col, 1.0)) if volr_col else pd.Series([0.5]*len(df), index=df.index)
    x_amt = _robust_minmax(_to_float_series(df, amt_col, 0.0)) if amt_col else pd.Series([0.5]*len(df), index=df.index)
    x_tr  = _robust_minmax(_to_float_series(df, tr_col, 0.0))  if tr_col  else pd.Series([0.5]*len(df), index=df.index)

    # Weighting: pct/strength prioritized, then volume/amount, then turnover
    strength01 = 0.40 * x_pct + 0.25 * x_vol + 0.25 * x_amt + 0.10 * x_tr
    strength01 = _clip(strength01, 0.0, 1.0)
    strength = strength01 * 100.0  # for readability

    # theme boost (optional)
    theme_boost_col = _first_existing_col(df, ["theme_boost", "_theme_boost", "ThemeBoost", "themeScore"])
    if theme_boost_col:
        theme_boost = _to_float_series(df, theme_boost_col, 1.0).replace([np.inf, -np.inf], 1.0)
        # keep sane range
        theme_boost = theme_boost.clip(0.0, 10.0).fillna(1.0)
    else:
        theme_boost = pd.Series([1.0] * len(df), index=df.index, dtype="float64")

    # board
    board = _ensure_board(df)

    # light risk handling (no hard gate)
    st_col = _first_existing_col(df, ["is_st", "st", "ST"])
    if st_col:
        is_st = _to_float_series(df, st_col, 0.0)
        st_penalty = np.where(is_st > 0.5, 0.85, 1.0)  # ST penalize
    else:
        st_penalty = 1.0

    # final score
    score = prob * strength * theme_boost
    if isinstance(st_penalty, np.ndarray):
        score = score * st_penalty

    # attach debug columns
    out_df = df.copy()
    out_df["_prob"] = prob
    out_df["_strength"] = strength
    out_df["_theme_boost"] = theme_boost
    out_df["_score"] = score
    out_df["_board"] = board

    # select topN
    settings = _get_step6_settings(s)
    topn = int(settings.topn or 10)
    topn = max(1, topn)

    top = out_df.sort_values(["_score", "_prob"], ascending=False).head(topn).copy()

    result = pd.DataFrame(
        {
            "ts_code": top[ts_col].astype(str),
            "name": top[name_col].astype(str),
            "score": pd.to_numeric(top["_score"], errors="coerce").fillna(0.0).astype(float).round(6),
            "prob": pd.to_numeric(top["_prob"], errors="coerce").fillna(0.0).astype(float).round(6),
            "board": top["_board"].astype(str),
            # debug / explain
            "StrengthScore": pd.to_numeric(top["_strength"], errors="coerce").fillna(0.0).astype(float).round(3),
            "ThemeBoost": pd.to_numeric(top["_theme_boost"], errors="coerce").fillna(1.0).astype(float).round(6),
        }
    ).reset_index(drop=True)

    return result


# Backward-compatible alias if your pipeline imports a different function name
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step6_final_topn(df, s=s)
