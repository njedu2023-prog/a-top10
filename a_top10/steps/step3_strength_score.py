from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# 通用工具（统一所有 Step 的数值处理方式）
# =========================================================
def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _to_float_series(df: pd.DataFrame, col: Optional[str], default=0.0) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce")
    return s.fillna(default).astype("float64")


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(0.0, 1.0)


def _robust_minmax(s: pd.Series) -> pd.Series:
    """抗异常值的 0-1 归一化"""
    if s.empty:
        return pd.Series(0.0, index=s.index, dtype="float64")

    q1, q99 = s.quantile([0.01, 0.99])
    if not np.isfinite(q1) or not np.isfinite(q99) or q99 <= q1:
        return pd.Series(0.0, index=s.index, dtype="float64")

    out = (s - q1) / (q99 - q1)
    return _clip01(out)


def _logistic01(x: pd.Series, center: float, scale: float) -> pd.Series:
    """平滑 S 型映射"""
    z = (x - center) / max(scale, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


def _normalize_turnover_rate(tr: pd.Series) -> pd.Series:
    """
    自动识别：
    - 0~1 → 转 %
    - 0~100 → 保持
    """
    med = float(tr.replace([np.inf, -np.inf], np.nan).dropna().median()) if len(tr) else np.nan
    if np.isfinite(med) and med <= 1.5:
        tr = tr * 100.0
    return tr


# =========================================================
# 核心评分函数
# =========================================================
def calc_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # -------------------------
    # 1️⃣ 动量（涨跌幅）
    # -------------------------
    pct_col = _first_existing_col(out, ["pct_change", "pct_chg", "涨跌幅"])
    pct = _to_float_series(out, pct_col, default=0.0)

    score_mom = _robust_minmax(pct)
    out["score_momentum"] = score_mom

    # -------------------------
    # 2️⃣ 成交额
    # -------------------------
    amt_col = _first_existing_col(out, ["amount", "成交额", "turnover_amount"])
    amt = _to_float_series(out, amt_col, default=0.0).clip(0.0)

    score_amt = _robust_minmax(np.log1p(amt))
    out["score_amount"] = score_amt

    # -------------------------
    # 3️⃣ 资金净流入
    # -------------------------
    net_col = _first_existing_col(out, ["net_amount", "净额", "净买入额"])
    net = _to_float_series(out, net_col, default=0.0)

    score_net = _robust_minmax(net)
    out["score_netflow"] = score_net

    # -------------------------
    # 4️⃣ 换手率
    # -------------------------
    tr_col = _first_existing_col(out, ["turnover_rate", "turn_rate", "换手率"])
    tr = _to_float_series(out, tr_col, default=5.0)
    tr = _normalize_turnover_rate(tr).clip(0.0, 100.0)

    score_turn = pd.Series(0.0, index=out.index)

    m1 = (tr > 2) & (tr <= 10)
    score_turn.loc[m1] = (tr.loc[m1] - 2) / 8

    m2 = (tr > 10) & (tr <= 25)
    score_turn.loc[m2] = 1 - 0.6 * ((tr.loc[m2] - 10) / 15)

    m3 = tr > 25
    score_turn.loc[m3] = 0.4 * (1 - ((tr.loc[m3] - 25) / 35)).clip(0, 1)

    out["score_turnover"] = _clip01(score_turn)

    # -------------------------
    # 5️⃣ 龙虎榜
    # -------------------------
    l_amt_col = _first_existing_col(out, ["l_amount", "龙虎榜成交额"])
    l_amt = _to_float_series(out, l_amt_col, default=0.0).clip(0)

    score_lhb_amt = _robust_minmax(np.log1p(l_amt))

    rate_col = _first_existing_col(out, ["amount_rate", "成交额占比"])
    if rate_col:
        ar = _to_float_series(out, rate_col, default=0.0)
        med = float(ar.replace([np.inf, -np.inf], np.nan).dropna().median())
        if np.isfinite(med) and med <= 1.5:
            ar *= 100
        score_lhb_rate = _logistic01(ar, center=10, scale=10)
    else:
        score_lhb_rate = pd.Series(0.0, index=out.index)

    out["score_lhb"] = _clip01(0.6 * score_lhb_amt + 0.4 * score_lhb_rate)

    # -------------------------
    # 6️⃣ 市值（越小越强）
    # -------------------------
    fv_col = _first_existing_col(out, ["float_values", "流通市值", "float_mv"])
    if fv_col:
        fv = _to_float_series(out, fv_col, default=np.nan).clip(0)
        size01 = _robust_minmax(np.log1p(fv.fillna(0)))
        out["score_size"] = 1 - size01
    else:
        out["score_size"] = 0.5

    # -------------------------
    # ⭐ 总强度
    # -------------------------
    out["StrengthScore"] = (
        0.30 * out["score_momentum"]
        + 0.22 * out["score_amount"]
        + 0.18 * out["score_netflow"]
        + 0.16 * out["score_turnover"]
        + 0.10 * out["score_lhb"]
        + 0.04 * out["score_size"]
    ) * 100.0

    return out.sort_values("StrengthScore", ascending=False)


# =========================================================
# Runner（保持接口不变）
# =========================================================
def run_step3(candidates_df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    scored = calc_strength_score(candidates_df)
    top_k = max(1, int(top_k or 50))
    return scored.head(top_k)


def run(df: pd.DataFrame, s=None, top_k: int = 50) -> pd.DataFrame:
    return run_step3(df, top_k=top_k)


if __name__ == "__main__":
    print("Step3 StrengthScore module loaded.")
