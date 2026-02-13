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
    s = s.fillna(default)
    return s


def _clip01(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(x, dtype="float64"), 0.0, 1.0), index=getattr(x, "index", None))


def _winsorize(s: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    """分位截尾，减少极端值影响"""
    if len(s) == 0:
        return s
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    return s.clip(lo, hi)


def _robust_minmax(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    """
    稳健 MinMax：先按分位截尾，再映射到 0~1
    - 对长尾更稳健
    """
    s2 = _winsorize(s, lower_q, upper_q)
    lo = float(s2.min())
    hi = float(s2.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series([0.0] * len(s), index=s.index, dtype="float64")
    return (s2 - lo) / (hi - lo)


def _logistic01(s: pd.Series, center: float, scale: float) -> pd.Series:
    """
    logistic 映射到 0~1：
    - center: 中心点
    - scale: 越小越陡（需要 >0）
    """
    scale = max(float(scale), 1e-9)
    x = (s - float(center)) / scale
    # 防溢出保护
    x = x.clip(-60, 60)
    return pd.Series(1.0 / (1.0 + np.exp(-x)), index=s.index, dtype="float64")


def _normalize_turnover_rate(s: pd.Series) -> pd.Series:
    """
    将换手率转为百分比口径（如果本来是 0~1 小数则 *100）
    """
    s = s.copy()
    med = float(s.replace([np.inf, -np.inf], np.nan).dropna().median()) if len(s) else np.nan
    if np.isfinite(med) and med <= 1.5:  # 常见：0.03 表示 3%
        s = s * 100.0
    return s


# -------------------------
# Core scoring
# -------------------------

def calc_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    核心思想（科学/合理的启发式）：
    1) “动量/强势”：涨跌幅越强越好（尤其是强阳线/涨停附近）
    2) “成交/关注度”：成交额越大越好（但做对数+稳健归一化）
    3) “资金净流”：净额/净占比越正越好（稳健映射）
    4) “换手结构”：过低=没流动性，过高=可能分歧大；偏好中高但不过热（用钟形/分段方式）
    5) “龙虎榜强度”：若有 l_amount / amount_rate 等，作为强化项
    6) “规模惩罚（轻微）”：流通市值过大通常弹性小（可选、权重小）
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    
    # --- FIX: normalize required identity columns for downstream steps (step4 needs them) ---

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

    # ---------- A1: Momentum / pct_change ----------
    pct_col = _first_existing_col(out, ["pct_change", "change_pct", "pct_chg", "涨跌幅"])
    pct = _to_float_series(out, pct_col, default=0.0)

    # 经验：A股日内强势通常在 3%~10%+，用 logistic 更平滑
    # center=3, scale=3 => 3% 左右为中性，>8% 接近高分
    score_momo = _logistic01(pct, center=3.0, scale=3.0)

    out["pct_change"] = pct
    out["score_momentum"] = score_momo

    # ---------- A2: Liquidity / amount ----------
    amt_col = _first_existing_col(out, ["amount", "成交额", "turnover_amount", "amt"])
    amount = _to_float_series(out, amt_col, default=0.0).clip(0.0, float("inf"))

    # 对数压缩 + 稳健归一化（成交额常长尾）
    log_amt = np.log1p(amount)
    score_amt = _robust_minmax(pd.Series(log_amt, index=out.index))

    out["amount"] = amount
    out["score_amount"] = score_amt

    # ---------- A3: Net flow / net_amount or net_rate ----------
    net_amt_col = _first_existing_col(out, ["net_amount", "net_amt", "净额", "净买入额"])
    net_rate_col = _first_existing_col(out, ["net_rate", "净占比", "net_ratio"])

    net_amt = _to_float_series(out, net_amt_col, default=0.0)
    net_rate = _to_float_series(out, net_rate_col, default=np.nan)

    # 资金净流：优先 net_rate（更可比），否则用 net_amount（做尺度压缩）
    if net_rate_col:
        # net_rate 通常是百分比（可能 -xx ~ +xx）
        score_net = _logistic01(net_rate, center=0.0, scale=5.0)  # 0% 中性，>10% 很强
        out["net_rate"] = net_rate
    else:
        # net_amount：对数压缩并保留符号（正好，负差）
        signed = np.sign(net_amt) * np.log1p(np.abs(net_amt))
        score_net = _logistic01(pd.Series(signed, index=out.index), center=0.0, scale=2.0)
        out["net_amount"] = net_amt

    out["score_netflow"] = score_net

    # ---------- A4: Turnover structure / turnover_rate ----------
    tr_col = _first_existing_col(out, ["turnover_rate", "turn_rate", "换手率", "turnover"])
    tr = _normalize_turnover_rate(_to_float_series(out, tr_col, default=5.0)).clip(0.0, 100.0)
    out["turnover_rate"] = tr

    # 偏好：中高换手（例如 6%~20%），过低/过高扣分
    # 用“分段三角形”更直观稳健
    #  - <=2% 低分
    #  - 2%~10% 上升到高分
    #  - 10%~25% 缓慢下降
    #  - >25% 逐步扣到低分
    score_turn = pd.Series(0.0, index=out.index, dtype="float64")
    t = tr

    # 上升段 2~10
    m1 = (t > 2.0) & (t <= 10.0)
    score_turn[m1] = (t[m1] - 2.0) / (10.0 - 2.0)

    # 平台/缓降 10~25：从 1 降到 0.4
    m2 = (t > 10.0) & (t <= 25.0)
    score_turn[m2] = 1.0 - 0.6 * ((t[m2] - 10.0) / (25.0 - 10.0))

    # 超高 >25：从 0.4 继续降到 0（到 60% 视为极端）
    m3 = t > 25.0
    score_turn[m3] = 0.4 * (1.0 - ((t[m3] - 25.0) / (60.0 - 25.0))).clip(0.0, 1.0)

    # 极低 <=2：保持 0
    out["score_turnover"] = _clip01(score_turn)

    # ---------- A5: LHB strength (optional) ----------
    # 龙虎榜成交额/成交额占比：存在则加强
    l_amt_col = _first_existing_col(out, ["l_amount", "龙虎榜成交额", "lhb_amount"])
    amt_rate_col = _first_existing_col(out, ["amount_rate", "成交额占比", "lhb_amount_rate"])

    l_amt = _to_float_series(out, l_amt_col, default=0.0).clip(0.0, float("inf"))
    amt_rate = _to_float_series(out, amt_rate_col, default=np.nan)

    # l_amount：对数+稳健归一化
    score_lhb_amt = _robust_minmax(pd.Series(np.log1p(l_amt), index=out.index))

    # amount_rate：通常 0~100 或 0~1，统一到 0~100 再 logistic
    if amt_rate_col:
        ar = amt_rate.copy()
        med_ar = float(ar.replace([np.inf, -np.inf], np.nan).dropna().median()) if len(ar) else np.nan
        if np.isfinite(med_ar) and med_ar <= 1.5:
            ar = ar * 100.0
        score_lhb_rate = _logistic01(ar.fillna(0.0), center=10.0, scale=10.0)  # 10% 左右中性
        out["amount_rate"] = ar
    else:
        score_lhb_rate = pd.Series(0.0, index=out.index, dtype="float64")

    out["l_amount"] = l_amt
    out["score_lhb"] = _clip01(0.6 * score_lhb_amt + 0.4 * score_lhb_rate)

    # ---------- A6: Size penalty (optional, small weight) ----------
    fv_col = _first_existing_col(out, ["float_values", "流通市值", "float_mv"])
    fv = _to_float_series(out, fv_col, default=np.nan).clip(0.0, float("inf"))

    # 市值越大弹性越小：用对数后做“反向得分”（但权重很小，避免误伤）
    if fv_col:
        log_fv = pd.Series(np.log1p(fv.fillna(0.0)), index=out.index)
        size01 = _robust_minmax(log_fv)          # 大市值 -> 1
        score_size = 1.0 - size01                # 大市值 -> 0
        out["float_values"] = fv
        out["score_size"] = _clip01(score_size)
    else:
        out["score_size"] = 0.5  # 没有就给中性常数，避免影响

    # ---------- Final aggregation ----------
    # 权重解释：
    # - momentum:   0.30  (强势/动量)
    # - amount:     0.22  (关注度/流动性)
    # - netflow:    0.18  (资金净流)
    # - turnover:   0.16  (结构健康)
    # - lhb:        0.10  (龙虎榜强化，可选)
    # - size:       0.04  (轻微规模因子)
    out["StrengthScore"] = (
        0.30 * out["score_momentum"]
        + 0.22 * out["score_amount"]
        + 0.18 * out["score_netflow"]
        + 0.16 * out["score_turnover"]
        + 0.10 * out["score_lhb"]
        + 0.04 * out["score_size"]
    ) * 100.0

    # 排序：StrengthScore 高者优先
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


# Backward-compatible alias：主程序若统一用 run() 调用每一步，这里也给出
def run(df: pd.DataFrame, s=None, top_k: int = 50) -> pd.DataFrame:
    return run_step3(df, top_k=top_k)


if __name__ == "__main__":
    print("Step3 (StrengthScore scheme A) module loaded successfully.")
