#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step3: Strength Score (方案A)
- 只依赖“现有字段”（来自候选/榜单/快照中常见的列）
- 输出 StrengthScore（0~100），并保留若干分项得分便于调试

推荐字段（存在则用，不存在则自动降级）：
- pct_change / change_pct / pct_chg / 涨跌幅
- turnover_rate / turn_rate / turnover / 换手率
- amount / 成交额
- net_amount / 净额
- net_rate / 净占比
- l_amount / 龙虎榜成交额
- amount_rate / 成交额占比
- float_values / 流通市值
"""

from __future__ import annotations

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
    """把输入压到 0~1，保持 index（如果有）"""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype="float64", copy=False)
        return pd.Series(np.clip(arr, 0.0, 1.0), index=x.index, dtype="float64")
    arr = np.asarray(x, dtype="float64")
    return pd.Series(np.clip(arr, 0.0, 1.0), dtype="float64")


def _winsorize(s: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    """分位截尾，减少极端值影响"""
    if s is None or len(s) == 0:
        return s
    s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s2) == 0:
        return s.fillna(0.0)
    lo = float(s2.quantile(lower_q))
    hi = float(s2.quantile(upper_q))
    return s.clip(lo, hi)


def _robust_minmax(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    """
    稳健 MinMax：先按分位截尾，再映射到 0~1
    - 对长尾更稳健
    """
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
    """
    logistic 映射到 0~1：
    - center: 中心点
    - scale: 越小越陡（需要 >0）
    """
    scale = max(float(scale), 1e-9)
    x = (s.astype("float64") - float(center)) / scale
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-60, 60)
    return pd.Series(1.0 / (1.0 + np.exp(-x.to_numpy(dtype="float64"))), index=s.index, dtype="float64")


def _normalize_turnover_rate(s: pd.Series) -> pd.Series:
    """
    将换手率转为百分比口径（如果本来是 0~1 小数则 *100）
    """
    s = s.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    med = float(s.median()) if len(s) else np.nan
    if np.isfinite(med) and med <= 1.5:  # 常见：0.03 表示 3%
        s = s * 100.0
    return s


def _ensure_identity_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    修复/补齐关键身份列，保证后续 Step4/5 可用：
    - ts_code
    - name
    - industry
    """
    if out is None or len(out) == 0:
        return out

    # 1) ts_code
    if "ts_code" not in out.columns or out["ts_code"].isna().all():
        for c in ["代码", "code", "symbol", "证券代码"]:
            if c in out.columns and not out[c].isna().all():
                out["ts_code"] = out[c].astype(str)
                break

    # 2) name
    if "name" not in out.columns or out["name"].isna().all():
        for c in ["股票", "名称", "stock_name"]:
            if c in out.columns and not out[c].isna().all():
                out["name"] = out[c].astype(str)
                break

    # 3) industry
    if "industry" not in out.columns or out["industry"].isna().all():
        for c in ["行业", "板块", "industry_name"]:
            if c in out.columns and not out[c].isna().all():
                out["industry"] = out[c].astype(str)
                break

    return out


# -------------------------
# Core scoring
# -------------------------

def calc_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    核心思想（科学/合理的启发式）：
    1) “动量/强势”：涨跌幅越强越好（尤其是强阳线/涨停附近）
    2) “成交/关注度”：成交额越大越好（但做对数+稳健归一化）
    3) “资金净流”：净额/净占比越正越好（稳健映射）
    4) “换手结构”：过低=没流动性，过高=可能分歧大；偏好中高但不过热（用分段方式）
    5) “龙虎榜强度”：若有 l_amount / amount_rate 等，作为强化项
    6) “规模惩罚（轻微）”：流通市值过大通常弹性小（可选、权重小）
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # ✅ 关键：先补齐身份列（避免 Step4/5 缺主键）
    out = _ensure_identity_columns(out)

    # ---------- A1: Momentum / pct_change ----------
    pct_col = _first_existing_col(out, ["pct_change", "change_pct", "pct_chg", "涨跌幅"])
    pct = _to_float_series(out, pct_col, default=0.0)

    score_momo = _logistic01(pct, center=3.0, scale=3.0)
    out["pct_change"] = pct
    out["score_momentum"] = score_momo

    # ---------- A2: Liquidity / amount ----------
    amt_col = _first_existing_col(out, ["amount", "成交额", "turnover_amount", "amt"])
    amount = _to_float_series(out, amt_col, default=0.0).clip(0.0, float("inf"))

    log_amt = pd.Series(np.log1p(amount.to_numpy(dtype="float64")), index=out.index, dtype="float64")
    score_amt = _robust_minmax(log_amt)

    out["amount"] = amount
    out["score_amount"] = score_amt

    # ---------- A3: Net flow / net_amount or net_rate ----------
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

    # ---------- A4: Turnover structure / turnover_rate ----------
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

    # ---------- A5: LHB strength (optional) ----------
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

    # ---------- A6: Size penalty (optional, small weight) ----------
    fv_col = _first_existing_col(out, ["float_values", "流通市值", "float_mv"])
    fv = _to_float_series(out, fv_col, default=np.nan).clip(0.0, float("inf"))

    if fv_col:
        log_fv = pd.Series(np.log1p(fv.fillna(0.0).to_numpy(dtype="float64")), index=out.index, dtype="float64")
        size01 = _robust_minmax(log_fv)      # 大市值 -> 1
        score_size = 1.0 - size01            # 大市值 -> 0
        out["float_values"] = fv
        out["score_size"] = _clip01(score_size)
    else:
        out["score_size"] = 0.5

    # ---------- Final aggregation ----------
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
    """
    返回 StrengthScore TopK 股票进入下一层
    """
    scored = calc_strength_score(candidates_df)
    top_k = int(top_k or 50)
    top_k = max(1, top_k)
    if len(scored) > top_k:
        scored = scored.head(top_k)
    return scored


def run(df: pd.DataFrame, s=None, top_k: int = 50) -> pd.DataFrame:
    """Backward-compatible alias"""
    return run_step3(df, top_k=top_k)


if __name__ == "__main__":
    print("Step3 (StrengthScore scheme A) module loaded successfully.")
