#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step3: Strength Score (方案A)
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -------------------------
# Utils
# -------------------------

def _first_existing_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return s.fillna(default)


def _clip01(x) -> pd.Series:
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype="float64", copy=False)
        return pd.Series(np.clip(arr, 0.0, 1.0), index=x.index, dtype="float64")
    arr = np.asarray(x, dtype="float64")
    return pd.Series(np.clip(arr, 0.0, 1.0), dtype="float64")


def _winsorize(s: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s2) == 0:
        return s.fillna(0.0)
    lo = float(s2.quantile(lower_q))
    hi = float(s2.quantile(upper_q))
    return s.clip(lo, hi)


def _robust_minmax(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series([], dtype="float64")

    s = s.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s2 = _winsorize(s, lower_q, upper_q)

    lo = float(s2.min())
    hi = float(s2.max())
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return pd.Series([0.0] * len(s2), index=s2.index, dtype="float64")
    return (s2 - lo) / (hi - lo)


def _logistic01(s: pd.Series, center: float, scale: float) -> pd.Series:
    scale = max(float(scale), 1e-9)
    x = (s.astype("float64") - float(center)) / scale
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-60, 60)
    return pd.Series(1.0 / (1.0 + np.exp(-x.to_numpy(dtype="float64"))), index=s.index, dtype="float64")


def _normalize_turnover_rate(s: pd.Series) -> pd.Series:
    s = s.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    med = float(s.median()) if len(s) else np.nan
    if np.isfinite(med) and med <= 1.5:
        s = s * 100.0
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


_TS_LIKE_RE = re.compile(r"\d{6}")


def _looks_like_ts_code_series(sr: pd.Series) -> bool:
    if sr is None or len(sr) == 0:
        return False
    ss = sr.astype(str).map(lambda x: x.strip())
    # 认为“至少有一半”看起来像代码
    return bool((ss.str.contains(_TS_LIKE_RE)).mean() >= 0.5)


def _ensure_identity_columns(out: pd.DataFrame) -> pd.DataFrame:
    if out is None or len(out) == 0:
        return out

    for key in ["ts_code", "name", "industry"]:
        if key in out.columns:
            out[key] = _as_str_series(out, key)

    # 0) 兜底：若 ts_code 列为空/不存在，但 index 像代码，把 index 提取出来
    if ("ts_code" not in out.columns) or _is_effectively_empty(out["ts_code"]):
        idx_s = pd.Series(out.index, index=out.index, dtype="object")
        idx_s = idx_s.where(~pd.isna(idx_s), pd.NA)
        if _looks_like_ts_code_series(idx_s.astype(str)):
            out["ts_code"] = idx_s.astype(str).map(lambda x: x.strip() if isinstance(x, str) else x)

    # 1) ts_code
    if ("ts_code" not in out.columns) or _is_effectively_empty(out["ts_code"]):
        for c in ["ts_code", "代码", "code", "symbol", "证券代码", "ticker"]:
            if c in out.columns:
                cand = _as_str_series(out, c)
                if not _is_effectively_empty(cand):
                    out["ts_code"] = cand
                    break

    # 2) name
    if ("name" not in out.columns) or _is_effectively_empty(out["name"]):
        for c in ["name", "股票", "名称", "stock_name", "证券名称"]:
            if c in out.columns:
                cand = _as_str_series(out, c)
                if not _is_effectively_empty(cand):
                    out["name"] = cand
                    break

    # 3) industry
    if ("industry" not in out.columns) or _is_effectively_empty(out["industry"]):
        for c in ["industry", "行业", "板块", "industry_name", "所属行业", "概念", "主题"]:
            if c in out.columns:
                cand = _as_str_series(out, c)
                if not _is_effectively_empty(cand):
                    out["industry"] = cand
                    break

    for key in ["ts_code", "name", "industry"]:
        if key in out.columns:
            out[key] = _as_str_series(out, key)

    return out


# -------------------------
# Core scoring
# -------------------------

def calc_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    out = _ensure_identity_columns(out)

    pct_col = _first_existing_col(out, ["pct_change", "change_pct", "pct_chg", "涨跌幅"])
    pct = _to_float_series(out, pct_col, default=0.0)
    score_momo = _logistic01(pct, center=3.0, scale=3.0)
    out["pct_change"] = pct
    out["score_momentum"] = score_momo

    amt_col = _first_existing_col(out, ["amount", "成交额", "turnover_amount", "amt"])
    amount = _to_float_series(out, amt_col, default=0.0).clip(0.0, float("inf"))
    log_amt = pd.Series(np.log1p(amount.to_numpy(dtype="float64")), index=out.index, dtype="float64")
    score_amt = _robust_minmax(log_amt)
    out["amount"] = amount
    out["score_amount"] = score_amt

    net_amt_col = _first_existing_col(out, ["net_amount", "net_amt", "净额", "净买入额"])
    net_rate_col = _first_existing_col(out, ["net_rate", "净占比", "net_ratio"])
    net_amt = _to_float_series(out, net_amt_col, default=0.0)
    net_rate = _to_float_series(out, net_rate_col, default=np.nan)

    if net_rate_col:
        score_net = _logistic01(net_rate.fillna(0.0), center=0.0, scale=5.0)
        out["net_rate"] = net_rate
    else:
        signed = np.sign(net_amt.to_numpy(dtype="float64")) * np.log1p(np.abs(net_amt.to_numpy(dtype="float64")))
        signed_s = pd.Series(signed, index=out.index, dtype="float64")
        score_net = _logistic01(signed_s, center=0.0, scale=2.0)
        out["net_amount"] = net_amt

    out["score_netflow"] = score_net

    tr_col = _first_existing_col(out, ["turnover_rate", "turn_rate", "换手率", "turnover"])
    tr = _to_float_series(out, tr_col, default=5.0)
    tr = _normalize_turnover_rate(tr).clip(0.0, 100.0)
    out["turnover_rate"] = tr

    score_turn = pd.Series(0.0, index=out.index, dtype="float64")
    t = tr
    m1 = (t > 2.0) & (t <= 10.0)
    score_turn.loc[m1] = (t.loc[m1] - 2.0) / (10.0 - 2.0)
    m2 = (t > 10.0) & (t <= 25.0)
    score_turn.loc[m2] = 1.0 - 0.6 * ((t.loc[m2] - 10.0) / (25.0 - 10.0))
    m3 = t > 25.0
    score_turn.loc[m3] = 0.4 * (1.0 - ((t.loc[m3] - 25.0) / (60.0 - 25.0))).clip(0.0, 1.0)
    out["score_turnover"] = _clip01(score_turn)

    l_amt_col = _first_existing_col(out, ["l_amount", "龙虎榜成交额", "lhb_amount"])
    amt_rate_col = _first_existing_col(out, ["amount_rate", "成交额占比", "lhb_amount_rate"])
    l_amt = _to_float_series(out, l_amt_col, default=0.0).clip(0.0, float("inf"))
    score_lhb_amt = _robust_minmax(pd.Series(np.log1p(l_amt.to_numpy(dtype="float64")), index=out.index, dtype="float64"))

    if amt_rate_col:
        ar = _to_float_series(out, amt_rate_col, default=0.0)
        med_ar = float(ar.replace([np.inf, -np.inf], np.nan).dropna().median()) if len(ar) else np.nan
        if np.isfinite(med_ar) and med_ar <= 1.5:
            ar = ar * 100.0
        score_lhb_rate = _logistic01(ar.fillna(0.0), center=10.0, scale=10.0)
        out["amount_rate"] = ar
    else:
        score_lhb_rate = pd.Series(0.0, index=out.index, dtype="float64")

    out["l_amount"] = l_amt
    out["score_lhb"] = _clip01(0.6 * score_lhb_amt + 0.4 * score_lhb_rate)

    fv_col = _first_existing_col(out, ["float_values", "流通市值", "float_mv"])
    fv = _to_float_series(out, fv_col, default=np.nan).clip(0.0, float("inf"))
    if fv_col:
        log_fv = pd.Series(np.log1p(fv.fillna(0.0).to_numpy(dtype="float64")), index=out.index, dtype="float64")
        size01 = _robust_minmax(log_fv)
        score_size = 1.0 - size01
        out["float_values"] = fv
        out["score_size"] = _clip01(score_size)
    else:
        out["score_size"] = 0.5

    out["StrengthScore"] = (
        0.30 * out["score_momentum"]
        + 0.22 * out["score_amount"]
        + 0.18 * out["score_netflow"]
        + 0.16 * out["score_turnover"]
        + 0.10 * out["score_lhb"]
        + 0.04 * out["score_size"]
    ) * 100.0

    out = out.sort_values("StrengthScore", ascending=False)
    return out


# -------------------------
# Runner
# -------------------------

def run_step3(candidates_df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    scored = calc_strength_score(candidates_df)
    top_k = int(top_k or 50)
    top_k = max(1, top_k)
    if len(scored) > top_k:
        scored = scored.head(top_k)
    return scored


def run(df: pd.DataFrame, s=None, top_k: int = 50) -> pd.DataFrame:
    return run_step3(df, top_k=top_k)


if __name__ == "__main__":
    print("Step3 (StrengthScore scheme A) module loaded successfully.")
