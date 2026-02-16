#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Final Selector (TopN + Full Ranking) — Quant/Production Edition

设计目标（专业版）：
1) 工业级容错：字段名不一致、缺失列、空白列、脏数据（NaN/inf/字符串数字）都能稳住
2) 可解释：输出包含分项、惩罚项、归一化项、以及诊断 meta/warnings
3) 可配置：settings(s) 支持 dict/对象属性，两套 key（大小写、别名）都可
4) 可复现：排序 tie-break 固化；可选稳定 hash-breaker
5) 量化常见做法：winsor/clip、min_prob 过滤、ST 惩罚、去重保留最优、score 模式（geo/add/mul）
6) 行业通用：不绑定某个数据源；字段识别尽量广泛

返回:
{
  "topN": DataFrame(精简 TopN)   # 推荐主 key
  "topn": DataFrame(精简 TopN)   # 向后兼容
  "full": DataFrame(完整排序全集)
  "meta": dict                  # 诊断信息（新增，不影响旧逻辑）
  "warnings": list[str]         # 告警（新增，不影响旧逻辑）
}

可选 settings(s) 参数（均可缺省）:
- topn / TOPN: int，默认 10
- score_mode: "geo" | "add" | "mul" （默认 "geo"）

# 核心权重：会自动归一化
- w_prob / W_PROB: float，默认 1.0
- w_strength / W_STRENGTH: float，默认 0.0
- w_theme / W_THEME: float，默认 0.0

# 过滤
- min_prob / MIN_PROB: float，默认 0.0  （低于该概率过滤）
- min_score / MIN_SCORE: float，默认 None（低于该最终分过滤）

# 惩罚 / 乘子
- st_penalty / ST_PENALTY: float，默认 0.85
- emotion_factor / EMOTION_FACTOR: float，默认 1.0

# 归一化/截断
- strength_scale / STRENGTH_SCALE: float，默认 100.0  （strength->0~1 的缩放分母）
- theme_cap / THEME_CAP: float，默认 1.3            （theme->0~1 的缩放分母）
- prob_floor / PROB_FLOOR: float，默认 1e-9          （geo 模式 log 的 floor）
- strength_floor / STRENGTH_FLOOR: float，默认 1e-9
- theme_floor / THEME_FLOOR: float，默认 1e-9

# winsor（可选，默认不启用）
- winsor_prob / WINSOR_PROB: tuple(low_q, high_q) or None
- winsor_strength / WINSOR_STRENGTH: tuple(low_q, high_q) or None
- winsor_theme / WINSOR_THEME: tuple(low_q, high_q) or None

# 去重
- dedup_by_ts_code / DEDUP_BY_TS_CODE: bool，默认 True
- dedup_keep / DEDUP_KEEP: "best"|"first" 默认 "best"
  - best: 按最终分最高保留
  - first: 按现有顺序保留

# 稳定 tie-break（可选）
- stable_hash_tiebreak / STABLE_HASH_TIEBREAK: bool 默认 True
  用 ts_code/name/board 生成稳定 hash，作为最后排序字段，保证完全可复现

字段自动识别（尽量兼容）：
- ts_code: ["ts_code","code","symbol","ticker","证券代码","代码",...]
- name: ["name","stock_name","名称","证券名称",...]
- board: ["board","industry","concept","theme","板块","行业","题材",...]
- prob: ["Probability","probability","prob","_prob","proba","p",...]
- strength: ["StrengthScore","strength","_strength",...]
- theme: ["ThemeBoost","theme","theme_boost","_theme",...]

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List, Union
import math
import hashlib

import numpy as np
import pandas as pd


# =========================
# Utilities
# =========================

def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _get_setting(s: Any, names: Sequence[str], default: Any) -> Any:
    """支持 dict / Mapping 与对象属性两种 settings。大小写/别名都兼容。"""
    if s is None:
        return default

    # dict / Mapping
    if _is_mapping(s):
        for k in names:
            if k in s:
                return s.get(k, default)
            lk = str(k).lower()
            if lk in s:
                return s.get(lk, default)
        return default

    # object attributes
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


def _coerce_df(obj: Any) -> pd.DataFrame:
    """
    把输入统一转换为 DataFrame：
    - DataFrame: 原样返回
    - dict: 从常见键名取 DataFrame；否则遍历 values 找第一个 DataFrame
    """
    if obj is None:
        return pd.DataFrame()

    if isinstance(obj, pd.DataFrame):
        return obj

    if _is_mapping(obj):
        for k in ("df", "data", "result", "full", "candidates", "candidate", "pool", "merged"):
            v = obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
        raise TypeError(
            f"Step6: got dict but cannot find DataFrame in common keys/values. keys={list(obj.keys())}"
        )

    raise TypeError(f"Step6 expects DataFrame or dict(containing DataFrame), got {type(obj)}")


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


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float) -> pd.Series:
    if (col is None) or (col not in df.columns):
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


def _clip01(x: pd.Series) -> pd.Series:
    return x.astype("float64").clip(0.0, 1.0)


def _winsorize(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    """winsorize（分位数截断），用于抗极端值；q_low/q_high in [0,1] 且 q_low < q_high"""
    if s is None or s.empty:
        return s
    q_low = float(q_low)
    q_high = float(q_high)
    if not (0.0 <= q_low < q_high <= 1.0):
        return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lo, hi)


def _stable_hash(*parts: str) -> int:
    """稳定 hash（跨进程/跨平台）用于排序最后 tie-break。"""
    text = "||".join([p if p is not None else "" for p in parts])
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    # 取 8 bytes
    return int(h[:16], 16)


def _toggle_empty(df: pd.DataFrame, col: str) -> pd.Series:
    """输出 TopN 时用：把 NA/None/'nan' 变成 ''，避免 markdown/csv 里难看。"""
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.where(~pd.isna(s), "")
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: "" if (x is None) or (isinstance(x, str) and x.lower() == "nan") else x)
    return s.astype("object")


# =========================
# Config
# =========================

@dataclass
class Step6Config:
    topn: int = 10
    score_mode: str = "geo"  # geo | add | mul

    w_prob: float = 1.0
    w_strength: float = 0.0
    w_theme: float = 0.0

    min_prob: float = 0.0
    min_score: Optional[float] = None

    st_penalty: float = 0.85
    emotion_factor: float = 1.0

    strength_scale: float = 100.0
    theme_cap: float = 1.3

    prob_floor: float = 1e-9
    strength_floor: float = 1e-9
    theme_floor: float = 1e-9

    winsor_prob: Optional[Tuple[float, float]] = None
    winsor_strength: Optional[Tuple[float, float]] = None
    winsor_theme: Optional[Tuple[float, float]] = None

    dedup_by_ts_code: bool = True
    dedup_keep: str = "best"  # best | first

    stable_hash_tiebreak: bool = True


def _load_config(s: Any) -> Step6Config:
    cfg = Step6Config()

    cfg.topn = int(_get_setting(s, ["topn", "TOPN"], cfg.topn) or cfg.topn)
    cfg.topn = max(1, cfg.topn)

    cfg.score_mode = str(_get_setting(s, ["score_mode", "SCORE_MODE"], cfg.score_mode) or cfg.score_mode).lower()
    if cfg.score_mode not in ("geo", "add", "mul"):
        cfg.score_mode = "geo"

    cfg.w_prob = float(_get_setting(s, ["w_prob", "W_PROB"], cfg.w_prob))
    cfg.w_strength = float(_get_setting(s, ["w_strength", "W_STRENGTH"], cfg.w_strength))
    cfg.w_theme = float(_get_setting(s, ["w_theme", "W_THEME"], cfg.w_theme))

    cfg.min_prob = float(_get_setting(s, ["min_prob", "MIN_PROB"], cfg.min_prob))
    cfg.min_prob = max(0.0, min(1.0, cfg.min_prob))

    ms = _get_setting(s, ["min_score", "MIN_SCORE"], cfg.min_score)
    cfg.min_score = None if ms is None else float(ms)

    cfg.st_penalty = float(_get_setting(s, ["st_penalty", "ST_PENALTY"], cfg.st_penalty))
    cfg.st_penalty = float(np.clip(cfg.st_penalty, 0.0, 1.0))

    cfg.emotion_factor = float(_get_setting(s, ["emotion_factor", "EMOTION_FACTOR"], cfg.emotion_factor))

    cfg.strength_scale = float(_get_setting(s, ["strength_scale", "STRENGTH_SCALE"], cfg.strength_scale))
    if cfg.strength_scale <= 0:
        cfg.strength_scale = 100.0

    cfg.theme_cap = float(_get_setting(s, ["theme_cap", "THEME_CAP"], cfg.theme_cap))
    if cfg.theme_cap <= 0:
        cfg.theme_cap = 1.3

    cfg.prob_floor = float(_get_setting(s, ["prob_floor", "PROB_FLOOR"], cfg.prob_floor))
    cfg.strength_floor = float(_get_setting(s, ["strength_floor", "STRENGTH_FLOOR"], cfg.strength_floor))
    cfg.theme_floor = float(_get_setting(s, ["theme_floor", "THEME_FLOOR"], cfg.theme_floor))
    cfg.prob_floor = max(1e-12, cfg.prob_floor)
    cfg.strength_floor = max(1e-12, cfg.strength_floor)
    cfg.theme_floor = max(1e-12, cfg.theme_floor)

    cfg.dedup_by_ts_code = bool(_get_setting(s, ["dedup_by_ts_code", "DEDUP_BY_TS_CODE"], cfg.dedup_by_ts_code))
    cfg.dedup_keep = str(_get_setting(s, ["dedup_keep", "DEDUP_KEEP"], cfg.dedup_keep) or cfg.dedup_keep).lower()
    if cfg.dedup_keep not in ("best", "first"):
        cfg.dedup_keep = "best"

    cfg.stable_hash_tiebreak = bool(_get_setting(
        s, ["stable_hash_tiebreak", "STABLE_HASH_TIEBREAK"], cfg.stable_hash_tiebreak
    ))

    # winsor tuples
    wp = _get_setting(s, ["winsor_prob", "WINSOR_PROB"], None)
    ws = _get_setting(s, ["winsor_strength", "WINSOR_STRENGTH"], None)
    wt = _get_setting(s, ["winsor_theme", "WINSOR_THEME"], None)

    def _parse_winsor(x) -> Optional[Tuple[float, float]]:
        if x is None:
            return None
        if isinstance(x, (list, tuple)) and len(x) == 2:
            try:
                a, b = float(x[0]), float(x[1])
                if 0.0 <= a < b <= 1.0:
                    return (a, b)
            except Exception:
                return None
        return None

    cfg.winsor_prob = _parse_winsor(wp)
    cfg.winsor_strength = _parse_winsor(ws)
    cfg.winsor_theme = _parse_winsor(wt)

    return cfg


# =========================
# Core Step6
# =========================

def run_step6_final_topn(df: Any, s=None) -> Dict[str, Any]:
    warnings: List[str] = []
    cfg = _load_config(s)

    df = _coerce_df(df)
    if df is None or df.empty:
        empty = pd.DataFrame()
        return {"topN": empty, "topn": empty, "full": empty, "meta": {"empty": True}, "warnings": warnings}

    out = df.copy()

    # -------- 1) Column identification --------
    ts_col = _first_existing_col(out, ["ts_code", "TS_CODE", "code", "symbol", "ticker", "证券代码", "代码"]) or "ts_code"
    name_col = _first_existing_col(out, ["name", "NAME", "stock_name", "名称", "证券名称", "股票简称"]) or "name"
    board_col = _first_existing_col(out, ["board", "BOARD", "industry", "INDUSTRY", "板块", "行业", "concept", "theme", "题材"]) or "board"

    if ts_col not in out.columns:
        out[ts_col] = pd.NA
        warnings.append(f"missing ts_code column; created '{ts_col}' as NA")
    if name_col not in out.columns:
        out[name_col] = pd.NA
        warnings.append(f"missing name column; created '{name_col}' as NA")
    if board_col not in out.columns:
        out[board_col] = pd.NA
        warnings.append(f"missing board/industry column; created '{board_col}' as NA")

    # normalize identity strings
    out[ts_col] = _as_str_series(out, ts_col)
    out[name_col] = _as_str_series(out, name_col)
    out[board_col] = _as_str_series(out, board_col)

    # fallback if identity columns are effectively empty
    if _is_effectively_empty(out[ts_col]):
        for c in ["ts_code", "TS_CODE", "证券代码", "代码", "code", "symbol", "ticker"]:
            if c in out.columns and not _is_effectively_empty(_as_str_series(out, c)):
                out[ts_col] = _as_str_series(out, c)
                warnings.append(f"ts_code fallback used from column '{c}'")
                break

    if _is_effectively_empty(out[name_col]):
        for c in ["name", "NAME", "名称", "证券名称", "stock_name", "股票简称"]:
            if c in out.columns and not _is_effectively_empty(_as_str_series(out, c)):
                out[name_col] = _as_str_series(out, c)
                warnings.append(f"name fallback used from column '{c}'")
                break

    # -------- 2) Read numeric factors --------
    prob_col = _first_existing_col(out, ["Probability", "probability", "prob", "_prob", "proba", "p", "预测概率"])
    str_col = _first_existing_col(out, ["StrengthScore", "strengthscore", "strength", "_strength", "强度", "强势度"])
    thm_col = _first_existing_col(out, ["ThemeBoost", "themeboost", "theme_boost", "theme", "_theme", "题材加成", "行业热度"])

    prob = _to_float_series(out, prob_col, 0.0)
    strength = _to_float_series(out, str_col, 0.0)
    theme = _to_float_series(out, thm_col, 1.0)

    # clean/clip to reasonable ranges
    prob = prob.clip(0.0, 1.0)
    strength = strength.clip(0.0, 1e9)  # 不预设上限，后续归一化 & winsor/scale
    theme = theme.clip(0.0, 1e9)

    # optional winsorization (quant-style)
    if cfg.winsor_prob is not None:
        prob = _winsorize(prob, cfg.winsor_prob[0], cfg.winsor_prob[1])
    if cfg.winsor_strength is not None:
        strength = _winsorize(strength, cfg.winsor_strength[0], cfg.winsor_strength[1])
    if cfg.winsor_theme is not None:
        theme = _winsorize(theme, cfg.winsor_theme[0], cfg.winsor_theme[1])

    out["_prob"] = prob.astype("float64")
    out["_strength"] = strength.astype("float64")
    out["_theme"] = theme.astype("float64")

    # -------- 3) ST detection & penalty --------
    st_col = _first_existing_col(out, ["is_st", "isST", "st", "st_flag", "ST", "风险ST"])
    if st_col is not None:
        st_series = _to_float_series(out, st_col, 0.0)
        st_flag = (st_series > 0.5).astype("float64")
    else:
        # 用 name 判定：*ST / ST（不使用捕获组，避免 pandas 警告）
        name_series = out[name_col].astype("object").fillna("").astype(str)
        st_flag = name_series.str.contains(r"^\*?ST|\bST\b", regex=True, na=False).astype("float64")

    st_penalty = pd.Series(
        np.where(st_flag.values > 0.5, cfg.st_penalty, 1.0),
        index=out.index,
        dtype="float64",
    )
    out["_st_flag"] = st_flag
    out["_st_penalty"] = st_penalty

    # -------- 4) Filtering --------
    if cfg.min_prob > 0:
        before = len(out)
        out = out[out["_prob"] >= cfg.min_prob].copy()
        after = len(out)
        if after == 0:
            empty = pd.DataFrame()
            warnings.append(f"all rows filtered out by min_prob={cfg.min_prob}")
            return {"topN": empty, "topn": empty, "full": empty, "meta": {"filtered_all": True}, "warnings": warnings}
        if after < before:
            warnings.append(f"filtered by min_prob={cfg.min_prob}: {before}->{after}")

    # -------- 5) Normalization (0..1) --------
    strength01 = _clip01(out["_strength"] / cfg.strength_scale)
    theme01 = _clip01(out["_theme"] / cfg.theme_cap)
    prob01 = _clip01(out["_prob"])

    out["_strength01"] = strength01
    out["_theme01"] = theme01
    out["_prob01"] = prob01

    # -------- 6) Weight normalization --------
    w_prob = float(cfg.w_prob)
    w_strength = float(cfg.w_strength)
    w_theme = float(cfg.w_theme)
    w_sum = max(1e-12, (w_prob + w_strength + w_theme))
    w_prob, w_strength, w_theme = w_prob / w_sum, w_strength / w_sum, w_theme / w_sum

    out["_w_prob"] = w_prob
    out["_w_strength"] = w_strength
    out["_w_theme"] = w_theme

    # -------- 7) Score computation --------
    if cfg.score_mode == "add":
        base_score = (w_prob * prob01 + w_strength * strength01 + w_theme * theme01)
    elif cfg.score_mode == "mul":
        base_score = (prob01 * strength01 * theme01)
    else:
        # geo mean in log space (robust)
        p = (prob01 + cfg.prob_floor)
        s01 = (strength01 + cfg.strength_floor)
        t01 = (theme01 + cfg.theme_floor)
        base_score = np.exp(
            (w_prob * np.log(p))
            + (w_strength * np.log(s01))
            + (w_theme * np.log(t01))
        )
        base_score = pd.Series(base_score, index=out.index, dtype="float64")

    final_score = base_score * out["_st_penalty"] * float(cfg.emotion_factor)

    out["_base_score"] = pd.Series(base_score, index=out.index, dtype="float64")
    out["_score"] = pd.Series(final_score, index=out.index, dtype="float64")
    out["_emotion_factor"] = float(cfg.emotion_factor)

    # optional min_score filter
    if cfg.min_score is not None:
        before = len(out)
        out = out[out["_score"] >= float(cfg.min_score)].copy()
        after = len(out)
        if after == 0:
            empty = pd.DataFrame()
            warnings.append(f"all rows filtered out by min_score={cfg.min_score}")
            return {"topN": empty, "topn": empty, "full": empty, "meta": {"filtered_all": True}, "warnings": warnings}
        if after < before:
            warnings.append(f"filtered by min_score={cfg.min_score}: {before}->{after}")

    # -------- 8) Dedup by ts_code --------
    if cfg.dedup_by_ts_code and ts_col in out.columns:
        if cfg.dedup_keep == "first":
            before = len(out)
            out = out.drop_duplicates(subset=[ts_col], keep="first").copy()
            after = len(out)
            if after < before:
                warnings.append(f"dedup_by_ts_code keep=first: {before}->{after}")
        else:
            before = len(out)
            # 先按分数/分项排序，再 drop_duplicates 保留最优
            out.sort_values(
                ["_score", "_prob", "_strength", "_theme"],
                ascending=False,
                inplace=True
            )
            out = out.drop_duplicates(subset=[ts_col], keep="first").copy()
            after = len(out)
            if after < before:
                warnings.append(f"dedup_by_ts_code keep=best: {before}->{after}")

    # -------- 9) Stable tie-break --------
    if cfg.stable_hash_tiebreak:
        # 用 ts_code/name/board 生成稳定 hash 作为最终 tie-break
        ts_s = out[ts_col].astype("object").fillna("").astype(str)
        nm_s = out[name_col].astype("object").fillna("").astype(str)
        bd_s = out[board_col].astype("object").fillna("").astype(str)
        out["_tiebreak_hash"] = [
            _stable_hash(ts_s.iat[i], nm_s.iat[i], bd_s.iat[i]) for i in range(len(out))
        ]
    else:
        out["_tiebreak_hash"] = 0

    # -------- 10) Full ranking sort (deterministic) --------
    full_sorted = out.sort_values(
        ["_score", "_prob", "_strength", "_theme", "_tiebreak_hash"],
        ascending=[False, False, False, False, True],
        kind="mergesort",  # 稳定排序
    ).copy()
    full_sorted.reset_index(drop=True, inplace=True)
    full_sorted["rank"] = full_sorted.index + 1

    # -------- 11) TopN output --------
    top_df_raw = full_sorted.head(cfg.topn).copy()

    top_df = pd.DataFrame({
        "rank": top_df_raw["rank"].astype(int),
        "ts_code": top_df_raw[ts_col].astype("object").fillna("").astype(str),
        "name": _toggle_empty(top_df_raw, name_col),
        "board": _toggle_empty(top_df_raw, board_col),

        # 最终分 + 分项
        "score": top_df_raw["_score"].round(6),
        "base_score": top_df_raw["_base_score"].round(6),

        "prob": top_df_raw["_prob"].round(6),
        "prob01": top_df_raw["_prob01"].round(6),

        "StrengthScore": top_df_raw["_strength"].round(3),
        "strength01": top_df_raw["_strength01"].round(6),

        "ThemeBoost": top_df_raw["_theme"].round(3),
        "theme01": top_df_raw["_theme01"].round(6),

        # 解释字段
        "st_flag": top_df_raw["_st_flag"].round(0).astype(int),
        "st_penalty": top_df_raw["_st_penalty"].round(3),
        "emotion_factor": top_df_raw["_emotion_factor"].round(6),
    })

    meta = {
        "rows_in": int(len(df)),
        "rows_out": int(len(full_sorted)),
        "topn": int(cfg.topn),
        "score_mode": cfg.score_mode,
        "weights": {"w_prob": w_prob, "w_strength": w_strength, "w_theme": w_theme},
        "filters": {"min_prob": cfg.min_prob, "min_score": cfg.min_score},
        "penalties": {"st_penalty": cfg.st_penalty, "emotion_factor": cfg.emotion_factor},
        "normalization": {"strength_scale": cfg.strength_scale, "theme_cap": cfg.theme_cap},
        "winsor": {
            "prob": cfg.winsor_prob, "strength": cfg.winsor_strength, "theme": cfg.winsor_theme
        },
        "dedup": {"enabled": cfg.dedup_by_ts_code, "keep": cfg.dedup_keep},
        "tiebreak": {"stable_hash_tiebreak": cfg.stable_hash_tiebreak},
        "columns_used": {"ts_col": ts_col, "name_col": name_col, "board_col": board_col,
                         "prob_col": prob_col, "strength_col": str_col, "theme_col": thm_col},
    }

    return {
        "topN": top_df,
        "topn": top_df,   # backward compatible
        "full": full_sorted,
        "meta": meta,
        "warnings": warnings,
    }


def run(df: Any, s=None) -> Dict[str, Any]:
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 FinalTopN Quant/Production Edition ready.")
