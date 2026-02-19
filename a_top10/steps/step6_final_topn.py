#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Final Selector (TopN + Full Ranking) — Quant/Production Edition (Enhanced)

【本版本新增兜底修复 + 报告表增强】
1) 若 Step6 输入 df 没带 Step3 的 StrengthScore（导致最终“强度得分=0”），将自动尝试从本地输出文件合并：
   outputs/step3_strength_{trade_date}.csv  或 outputs/step3_strength.csv（以及同目录备选）
2) 新增输出 limit_up_table（不破坏主返回结构）：
   - 用于报告《所有涨停股票的强度列表》字段对齐 Top10 表：
     排名序号 / Probability / 强度得分 / 题材加成 / 板块
   - 需要 Step6 的输入是 dict 且包含当日涨停列表 DataFrame（例如 keys: limit_list_d/limit_up/limit_df 等）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
import hashlib
from pathlib import Path

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

    if _is_mapping(s):
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


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    安全数值化，并返回诊断：
      - non_numeric_ratio: 原列中无法转成数值的占比（粗略）
      - used_default_ratio: 最终用 default 填充的占比
    """
    diag = {"col": col, "non_numeric_ratio": None, "used_default_ratio": None}
    if (col is None) or (col not in df.columns):
        s = pd.Series([default] * len(df), index=df.index, dtype="float64")
        diag["used_default_ratio"] = 1.0 if len(df) > 0 else 0.0
        return s, diag

    raw = df[col]
    num = pd.to_numeric(raw, errors="coerce")
    non_numeric = num.isna() & (~pd.isna(raw))
    if len(df) > 0:
        diag["non_numeric_ratio"] = float(non_numeric.mean())
    num = num.astype("float64")
    num = num.replace([np.inf, -np.inf], np.nan)

    used_default = num.isna()
    if len(df) > 0:
        diag["used_default_ratio"] = float(used_default.mean())
    num = num.fillna(default)

    return num, diag


def _clip01(x: pd.Series) -> pd.Series:
    return x.astype("float64").clip(0.0, 1.0)


def _winsorize(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    if s is None or s.empty:
        return s
    q_low = float(q_low)
    q_high = float(q_high)
    if not (0.0 <= q_low < q_high <= 1.0):
        return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lo, hi)


def _stable_hash_text(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _toggle_empty(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.where(~pd.isna(s), "")
    s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: "" if (x is None) or (isinstance(x, str) and x.lower() == "nan") else x)
    return s.astype("object")


def _nonzero_ratio(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return 0.0
    return float((x.astype("float64") != 0).mean())


def _detect_trade_date(df: pd.DataFrame, s: Any) -> Optional[str]:
    td = _get_setting(s, ["trade_date", "TRADE_DATE", "date", "DATE"], None)
    if td is not None:
        td = str(td).strip()
        if td:
            return td

    td_col = _first_existing_col(df, ["trade_date", "TRADE_DATE", "TradeDate", "date", "日期"])
    if td_col and td_col in df.columns:
        vals = df[td_col].astype("object").dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) > 0:
            return vals.value_counts().index[0]
    return None


def _find_limit_up_df(src: Optional[Mapping[str, Any]]) -> Optional[pd.DataFrame]:
    """
    从 Step6 输入 dict 中探测“当日涨停列表”DataFrame。
    兼容常见 key：
      limit_list_d / limit_list / limit_up / limitup / limit_df / limit / stk_limit / limit_break_d ...
    """
    if not src:
        return None

    keys = list(src.keys())
    prefer = [
        "limit_list_d", "limit_list", "limit_up", "limitup", "limit_df", "limit",
        "stk_limit", "limit_break_d", "limit_list", "limit_list_d.csv"
    ]

    # 1) 优先按候选 key 精确/忽略大小写匹配
    lower_map = {str(k).lower(): k for k in keys}
    for k in prefer:
        lk = str(k).lower()
        if lk in lower_map:
            v = src.get(lower_map[lk])
            if isinstance(v, pd.DataFrame) and not v.empty:
                return v

    # 2) 再遍历 values：列中含 ts_code 且看起来是涨停列表
    for v in src.values():
        if isinstance(v, pd.DataFrame) and not v.empty:
            tc = _first_existing_col(v, ["ts_code", "TS_CODE", "code", "symbol", "证券代码", "代码"])
            if tc:
                return v

    return None


def _try_enrich_strength_from_step3(
    out: pd.DataFrame,
    ts_col: str,
    s: Any,
    strength_present: bool,
    strength_nonzero_ratio: float,
    warnings: List[str],
):
    """
    如果 Step6 输入 df 没带强度列（或几乎全 0），自动从 outputs/step3_strength_{trade_date}.csv 合并回填。
    回填策略默认：仅对 out["_strength"]==0 的行，用外部 StrengthScore 覆盖。
    """
    meta = {
        "attempted": False,
        "used": False,
        "reason": "",
        "trade_date": None,
        "path": None,
        "rows_loaded": 0,
        "filled_count": 0,
        "mode": None,
    }

    force = bool(_get_setting(s, ["enrich_strength_force", "ENRICH_STRENGTH_FORCE"], False))
    threshold = float(_get_setting(s, ["enrich_strength_nonzero_threshold", "ENRICH_STRENGTH_NONZERO_THRESHOLD"], 0.01))
    fill_mode = str(_get_setting(s, ["enrich_strength_fill_mode", "ENRICH_STRENGTH_FILL_MODE"], "fill_zero")).lower()
    if fill_mode not in ("fill_zero", "overwrite"):
        fill_mode = "fill_zero"
    meta["mode"] = fill_mode

    should = force or (not strength_present) or (strength_nonzero_ratio < threshold)
    if not should:
        meta["reason"] = "skip: strength seems present and nonzero"
        return meta

    meta["attempted"] = True
    meta["reason"] = "force" if force else ("missing_strength" if not strength_present else f"nonzero_ratio<{threshold}")

    trade_date = _detect_trade_date(out, s)
    meta["trade_date"] = trade_date

    outputs_dir = _get_setting(s, ["outputs_dir", "OUTPUTS_DIR"], "outputs")
    outputs_dir = str(outputs_dir) if outputs_dir is not None else "outputs"

    candidates: List[Path] = []
    if trade_date:
        candidates.append(Path(outputs_dir) / f"step3_strength_{trade_date}.csv")
        candidates.append(Path("outputs") / f"step3_strength_{trade_date}.csv")
        candidates.append(Path(f"step3_strength_{trade_date}.csv"))

    candidates.append(Path(outputs_dir) / "step3_strength.csv")
    candidates.append(Path("outputs") / "step3_strength.csv")
    candidates.append(Path("step3_strength.csv"))

    target = None
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                target = p
                break
        except Exception:
            continue

    if target is None:
        warnings.append("strength enrich: could not find step3_strength csv locally (outputs/step3_strength_*.csv).")
        meta["used"] = False
        return meta

    meta["path"] = str(target)

    try:
        strength_df = pd.read_csv(target, dtype={"ts_code": "string"}, encoding="utf-8")
    except UnicodeDecodeError:
        strength_df = pd.read_csv(target, dtype={"ts_code": "string"}, encoding="utf-8-sig")
    except Exception as e:
        warnings.append(f"strength enrich: failed to read {target}: {e}")
        meta["used"] = False
        return meta

    meta["rows_loaded"] = int(len(strength_df))

    if strength_df.empty:
        warnings.append(f"strength enrich: {target} is empty.")
        meta["used"] = False
        return meta

    key_col = _first_existing_col(strength_df, ["ts_code", "TS_CODE", "code", "symbol", "ticker", "证券代码", "代码"])
    if key_col is None:
        warnings.append(f"strength enrich: {target} has no ts_code-like column.")
        meta["used"] = False
        return meta

    strength_col = _first_existing_col(strength_df, [
        "StrengthScore", "strengthscore", "_strength", "strength", "强度得分", "强度", "强势度"
    ])
    if strength_col is None:
        warnings.append(f"strength enrich: {target} has no StrengthScore-like column.")
        meta["used"] = False
        return meta

    tmp = strength_df[[key_col, strength_col]].copy()
    tmp.rename(columns={key_col: "__ts_code__", strength_col: "__strength_ext__"}, inplace=True)

    tmp["__ts_code__"] = tmp["__ts_code__"].astype("object").fillna("").astype(str).str.strip()
    tmp["__strength_ext__"] = pd.to_numeric(tmp["__strength_ext__"], errors="coerce").astype("float64")
    tmp["__strength_ext__"] = tmp["__strength_ext__"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out_key = out[ts_col].astype("object").fillna("").astype(str).str.strip()
    merged = out.copy()
    merged["__ts_code__"] = out_key

    merged = merged.merge(tmp, on="__ts_code__", how="left")

    ext = merged["__strength_ext__"].fillna(0.0).astype("float64")

    if "_strength" not in merged.columns:
        merged["_strength"] = 0.0

    cur = merged["_strength"].fillna(0.0).astype("float64")
    if fill_mode == "overwrite":
        new_strength = np.where(ext > 0, ext, cur)
    else:
        new_strength = np.where((cur <= 0) & (ext > 0), ext, cur)

    filled = int(np.sum((cur != new_strength)))
    merged["_strength"] = pd.Series(new_strength, index=merged.index, dtype="float64")

    merged.drop(columns=["__ts_code__", "__strength_ext__"], inplace=True, errors="ignore")

    meta["used"] = True
    meta["filled_count"] = filled

    if filled > 0:
        warnings.append(f"strength enrich: merged from {target}, filled {filled} rows.")
    else:
        warnings.append(f"strength enrich: merged from {target}, but nothing was filled (maybe already present).")

    return merged, meta


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

    strength_default: float = 0.0
    theme_default: float = 1.0

    winsor_prob: Optional[Tuple[float, float]] = None
    winsor_strength: Optional[Tuple[float, float]] = None
    winsor_theme: Optional[Tuple[float, float]] = None

    dedup_by_ts_code: bool = True
    dedup_keep: str = "best"  # best | first

    stable_hash_tiebreak: bool = True

    # enrich options
    enrich_strength_force: bool = False
    enrich_strength_nonzero_threshold: float = 0.01
    enrich_strength_fill_mode: str = "fill_zero"  # fill_zero | overwrite
    outputs_dir: str = "outputs"


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

    cfg.strength_default = float(_get_setting(s, ["strength_default", "STRENGTH_DEFAULT"], cfg.strength_default))
    cfg.theme_default = float(_get_setting(s, ["theme_default", "THEME_DEFAULT"], cfg.theme_default))

    cfg.dedup_by_ts_code = bool(_get_setting(s, ["dedup_by_ts_code", "DEDUP_BY_TS_CODE"], cfg.dedup_by_ts_code))
    cfg.dedup_keep = str(_get_setting(s, ["dedup_keep", "DEDUP_KEEP"], cfg.dedup_keep) or cfg.dedup_keep).lower()
    if cfg.dedup_keep not in ("best", "first"):
        cfg.dedup_keep = "best"

    cfg.stable_hash_tiebreak = bool(_get_setting(
        s, ["stable_hash_tiebreak", "STABLE_HASH_TIEBREAK"], cfg.stable_hash_tiebreak
    ))

    # enrich options
    cfg.enrich_strength_force = bool(_get_setting(s, ["enrich_strength_force", "ENRICH_STRENGTH_FORCE"], cfg.enrich_strength_force))
    cfg.enrich_strength_nonzero_threshold = float(_get_setting(
        s, ["enrich_strength_nonzero_threshold", "ENRICH_STRENGTH_NONZERO_THRESHOLD"], cfg.enrich_strength_nonzero_threshold
    ))
    cfg.enrich_strength_fill_mode = str(_get_setting(
        s, ["enrich_strength_fill_mode", "ENRICH_STRENGTH_FILL_MODE"], cfg.enrich_strength_fill_mode
    ) or cfg.enrich_strength_fill_mode).lower()
    if cfg.enrich_strength_fill_mode not in ("fill_zero", "overwrite"):
        cfg.enrich_strength_fill_mode = "fill_zero"

    cfg.outputs_dir = str(_get_setting(s, ["outputs_dir", "OUTPUTS_DIR"], cfg.outputs_dir) or cfg.outputs_dir)

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

    src_mapping: Optional[Mapping[str, Any]] = df if _is_mapping(df) else None

    df = _coerce_df(df)
    if df is None or df.empty:
        empty = pd.DataFrame()
        return {"topN": empty, "topn": empty, "full": empty, "meta": {"empty": True}, "warnings": warnings}

    out = df.copy()

    # -------- 1) Column identification --------
    ts_col = _first_existing_col(out, [
        "ts_code", "TS_CODE", "code", "symbol", "ticker", "证券代码", "代码", "股票代码"
    ]) or "ts_code"
    name_col = _first_existing_col(out, [
        "name", "NAME", "stock_name", "名称", "证券名称", "股票简称", "股票"
    ]) or "name"
    board_col = _first_existing_col(out, [
        "board", "BOARD", "industry", "INDUSTRY", "板块", "行业", "concept", "theme", "题材", "所属行业"
    ]) or "board"

    if ts_col not in out.columns:
        out[ts_col] = pd.NA
        warnings.append(f"missing ts_code column; created '{ts_col}' as NA")
    if name_col not in out.columns:
        out[name_col] = pd.NA
        warnings.append(f"missing name column; created '{name_col}' as NA")
    if board_col not in out.columns:
        out[board_col] = pd.NA
        warnings.append(f"missing board/industry column; created '{board_col}' as NA")

    out[ts_col] = _as_str_series(out, ts_col)
    out[name_col] = _as_str_series(out, name_col)
    out[board_col] = _as_str_series(out, board_col)

    if _is_effectively_empty(out[ts_col]):
        for c in ["ts_code", "TS_CODE", "证券代码", "代码", "股票代码", "code", "symbol", "ticker"]:
            if c in out.columns and not _is_effectively_empty(_as_str_series(out, c)):
                out[ts_col] = _as_str_series(out, c)
                warnings.append(f"ts_code fallback used from column '{c}'")
                break

    if _is_effectively_empty(out[name_col]):
        for c in ["name", "NAME", "名称", "证券名称", "stock_name", "股票简称", "股票"]:
            if c in out.columns and not _is_effectively_empty(_as_str_series(out, c)):
                out[name_col] = _as_str_series(out, c)
                warnings.append(f"name fallback used from column '{c}'")
                break

    if _is_effectively_empty(out[board_col]):
        for c in ["board", "BOARD", "industry", "INDUSTRY", "板块", "行业", "concept", "theme", "题材", "所属行业"]:
            if c in out.columns and not _is_effectively_empty(_as_str_series(out, c)):
                out[board_col] = _as_str_series(out, c)
                warnings.append(f"board fallback used from column '{c}'")
                break

    # -------- 2) Read numeric factors --------
    prob_col = _first_existing_col(out, [
        "Probability", "probability", "prob", "_prob", "proba", "p", "预测概率", "概率"
    ])
    str_col = _first_existing_col(out, [
        "StrengthScore", "strengthscore", "strength", "_strength",
        "强度得分", "强度", "强势度", "强度分", "强度评分"
    ])
    thm_col = _first_existing_col(out, [
        "ThemeBoost", "themeboost", "theme_boost", "theme", "_theme",
        "题材加成", "行业热度", "题材", "热度"
    ])

    prob, prob_diag = _to_float_series(out, prob_col, 0.0)
    strength, str_diag = _to_float_series(out, str_col, float(cfg.strength_default))
    theme, thm_diag = _to_float_series(out, thm_col, float(cfg.theme_default))

    prob = prob.clip(0.0, 1.0)
    strength = strength.clip(0.0, 1e12)
    theme = theme.clip(0.0, 1e12)

    if cfg.winsor_prob is not None:
        prob = _winsorize(prob, cfg.winsor_prob[0], cfg.winsor_prob[1])
    if cfg.winsor_strength is not None:
        strength = _winsorize(strength, cfg.winsor_strength[0], cfg.winsor_strength[1])
    if cfg.winsor_theme is not None:
        theme = _winsorize(theme, cfg.winsor_theme[0], cfg.winsor_theme[1])

    out["_prob"] = prob.astype("float64")
    out["_strength"] = strength.astype("float64")
    out["_theme"] = theme.astype("float64")

    strength_nonzero_ratio = _nonzero_ratio(out["_strength"])
    theme_nonzero_ratio = _nonzero_ratio(out["_theme"])
    strength_present = (str_col is not None)

    # -------- 2.5) Enrich strength from Step3 outputs if missing/mostly zero --------
    enrich_meta: Dict[str, Any] = {"attempted": False, "used": False}

    # 将 cfg 映射到 dict settings（确保 cfg 生效）
    if s is None:
        s2 = {
            "enrich_strength_force": cfg.enrich_strength_force,
            "enrich_strength_nonzero_threshold": cfg.enrich_strength_nonzero_threshold,
            "enrich_strength_fill_mode": cfg.enrich_strength_fill_mode,
            "outputs_dir": cfg.outputs_dir,
        }
    else:
        s2 = s

    force = bool(_get_setting(s2, ["enrich_strength_force", "ENRICH_STRENGTH_FORCE"], cfg.enrich_strength_force))
    threshold = float(_get_setting(
        s2, ["enrich_strength_nonzero_threshold", "ENRICH_STRENGTH_NONZERO_THRESHOLD"], cfg.enrich_strength_nonzero_threshold
    ))
    should_enrich = force or (not strength_present) or (strength_nonzero_ratio < threshold)

    if should_enrich:
        result = _try_enrich_strength_from_step3(
            out=out,
            ts_col=ts_col,
            s=s2,
            strength_present=strength_present,
            strength_nonzero_ratio=strength_nonzero_ratio,
            warnings=warnings,
        )
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], pd.DataFrame):
            out, enrich_meta = result[0], result[1]
            strength_nonzero_ratio = _nonzero_ratio(out["_strength"])
        elif isinstance(result, dict):
            enrich_meta = result

    # -------- Diagnostics warnings --------
    if str_col is None:
        warnings.append(f"strength column not found in input df; used strength_default={cfg.strength_default} (may be enriched).")
    else:
        if strength_nonzero_ratio == 0.0:
            warnings.append(f"strength column '{str_col}' exists but all zeros after parsing (or default filled).")
        if str_diag.get("non_numeric_ratio") is not None and str_diag["non_numeric_ratio"] > 0.2:
            warnings.append(f"strength column '{str_col}' has high non-numeric ratio: {str_diag['non_numeric_ratio']:.2%}")

    if thm_col is None:
        warnings.append(f"theme column not found; used theme_default={cfg.theme_default}")
    else:
        if theme_nonzero_ratio == 0.0:
            warnings.append(f"theme column '{thm_col}' exists but all zeros after parsing (or default filled).")
        if thm_diag.get("non_numeric_ratio") is not None and thm_diag["non_numeric_ratio"] > 0.2:
            warnings.append(f"theme column '{thm_col}' has high non-numeric ratio: {thm_diag['non_numeric_ratio']:.2%}")

    if prob_col is None:
        warnings.append("probability column not found; used default 0.0 (this will likely flatten ranking).")

    # -------- 3) ST detection & penalty --------
    st_col = _first_existing_col(out, ["is_st", "isST", "st", "st_flag", "ST", "风险ST", "是否ST"])
    if st_col is not None:
        st_series, _ = _to_float_series(out, st_col, 0.0)
        st_flag = (st_series > 0.5).astype("float64")
    else:
        name_series = out[name_col].astype("object").fillna("").astype(str)
        st_flag = (
            name_series.str.contains(r"^\*?ST", regex=True, na=False)
            | name_series.str.contains("退市", na=False)
        ).astype("float64")

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

    if cfg.score_mode in ("mul", "geo"):
        if w_strength > 0 and float((strength01 <= 0).mean()) > 0.9:
            warnings.append("strength01 is mostly 0 while w_strength>0; geo/mul may collapse scores. "
                            "Consider strength_default=1.0 or fix upstream strength / enrich.")
        if w_theme > 0 and float((theme01 <= 0).mean()) > 0.9:
            warnings.append("theme01 is mostly 0 while w_theme>0; geo/mul may collapse scores. "
                            "Consider theme_default=1.0 or fix upstream theme.")

    # -------- 7) Score computation --------
    if cfg.score_mode == "add":
        base_score = (w_prob * prob01 + w_strength * strength01 + w_theme * theme01)
    elif cfg.score_mode == "mul":
        base_score = (prob01 * strength01 * theme01)
    else:
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
            out.sort_values(
                ["_score", "_prob", "_strength", "_theme"],
                ascending=False,
                inplace=True,
                kind="mergesort",
            )
            out = out.drop_duplicates(subset=[ts_col], keep="first").copy()
            after = len(out)
            if after < before:
                warnings.append(f"dedup_by_ts_code keep=best: {before}->{after}")

    # -------- 9) Stable tie-break --------
    if cfg.stable_hash_tiebreak:
        ts_s = out[ts_col].astype("object").fillna("").astype(str)
        nm_s = out[name_col].astype("object").fillna("").astype(str)
        bd_s = out[board_col].astype("object").fillna("").astype(str)
        joined = (ts_s + "||" + nm_s + "||" + bd_s).tolist()
        out["_tiebreak_hash"] = [_stable_hash_text(x) for x in joined]
    else:
        out["_tiebreak_hash"] = 0

    # -------- 10) Full ranking sort --------
    full_sorted = out.sort_values(
        ["_score", "_prob", "_strength", "_theme", "_tiebreak_hash"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
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

        "score": top_df_raw["_score"].round(6),
        "base_score": top_df_raw["_base_score"].round(6),

        "prob": top_df_raw["_prob"].round(6),
        "prob01": top_df_raw["_prob01"].round(6),

        "StrengthScore": top_df_raw["_strength"].round(6),
        "strength01": top_df_raw["_strength01"].round(6),

        "ThemeBoost": top_df_raw["_theme"].round(6),
        "theme01": top_df_raw["_theme01"].round(6),

        "st_flag": top_df_raw["_st_flag"].round(0).astype(int),
        "st_penalty": top_df_raw["_st_penalty"].round(3),
        "emotion_factor": top_df_raw["_emotion_factor"].round(6),
    })

    # -------- 12) NEW: Build limit_up_table for report --------
    limit_up_table = pd.DataFrame()
    try:
        limit_df = _find_limit_up_df(src_mapping)
        if isinstance(limit_df, pd.DataFrame) and not limit_df.empty:
            # limit list columns
            lim_ts = _first_existing_col(limit_df, ["ts_code", "TS_CODE", "code", "symbol", "证券代码", "代码"]) or "ts_code"
            lim_nm = _first_existing_col(limit_df, ["name", "NAME", "stock_name", "名称", "证券名称", "股票简称", "股票"]) or None

            lim = limit_df.copy()
            if lim_ts not in lim.columns:
                lim[lim_ts] = pd.NA
            lim[lim_ts] = lim[lim_ts].astype("object").fillna("").astype(str).str.strip()

            # build index from full_sorted
            f = full_sorted.copy()
            f["_ts_key"] = f[ts_col].astype("object").fillna("").astype(str).str.strip()

            # keep only limit-up codes (intersection)
            codes = set(lim[lim_ts].tolist())
            f2 = f[f["_ts_key"].isin(codes)].copy()
            f2.reset_index(drop=True, inplace=True)
            if len(f2) > 0:
                f2["rank_limit"] = f2.index + 1

                # name fallback: prefer full_sorted name, else limit list name
                nm_full = f2[name_col].astype("object").fillna("").astype(str)
                if lim_nm and lim_nm in lim.columns:
                    lim_nm_s = lim[[lim_ts, lim_nm]].copy()
                    lim_nm_s[lim_ts] = lim_nm_s[lim_ts].astype("object").fillna("").astype(str).str.strip()
                    lim_nm_s.rename(columns={lim_ts: "_ts_key", lim_nm: "_name_lim"}, inplace=True)
                    f2 = f2.merge(lim_nm_s, on="_ts_key", how="left")
                    nm_lim = f2["_name_lim"].astype("object").fillna("").astype(str)
                    nm_show = np.where(nm_full != "", nm_full, nm_lim)
                else:
                    nm_show = nm_full

                # 输出为“报告表字段”
                limit_up_table = pd.DataFrame({
                    "排名": f2["rank_limit"].astype(int),
                    "代码": f2["_ts_key"].astype("object").fillna("").astype(str),
                    "股票": pd.Series(nm_show, index=f2.index).astype("object"),
                    "Probability": f2["_prob"].round(6),
                    "强度得分": f2["_strength"].round(6),
                    "题材加成": f2["_theme"].round(6),
                    "板块": f2[board_col].astype("object").fillna("").astype(str),
                })
            else:
                warnings.append("limit_up_table: limit list provided, but no intersection with step6 full_sorted (ts_code mismatch?).")
    except Exception as e:
        warnings.append(f"limit_up_table: failed to build: {e}")

    meta = {
        "rows_in": int(len(df)),
        "rows_out": int(len(full_sorted)),
        "topn": int(cfg.topn),
        "score_mode": cfg.score_mode,
        "weights": {"w_prob": w_prob, "w_strength": w_strength, "w_theme": w_theme},
        "filters": {"min_prob": cfg.min_prob, "min_score": cfg.min_score},
        "penalties": {"st_penalty": cfg.st_penalty, "emotion_factor": cfg.emotion_factor},
        "normalization": {"strength_scale": cfg.strength_scale, "theme_cap": cfg.theme_cap},
        "floors": {"prob_floor": cfg.prob_floor, "strength_floor": cfg.strength_floor, "theme_floor": cfg.theme_floor},
        "defaults": {"strength_default": cfg.strength_default, "theme_default": cfg.theme_default},
        "winsor": {"prob": cfg.winsor_prob, "strength": cfg.winsor_strength, "theme": cfg.winsor_theme},
        "dedup": {"enabled": cfg.dedup_by_ts_code, "keep": cfg.dedup_keep},
        "tiebreak": {"stable_hash_tiebreak": cfg.stable_hash_tiebreak},
        "columns_used": {
            "ts_col": ts_col, "name_col": name_col, "board_col": board_col,
            "prob_col": prob_col, "strength_col": str_col, "theme_col": thm_col
        },
        "parse_diag": {"prob": prob_diag, "strength": str_diag, "theme": thm_diag},
        "nonzero_ratio": {
            "strength": float(_nonzero_ratio(full_sorted["_strength"])),
            "theme": float(_nonzero_ratio(full_sorted["_theme"]))
        },
        "enrich_strength": enrich_meta,
        "limit_up_table_rows": int(len(limit_up_table)) if isinstance(limit_up_table, pd.DataFrame) else 0,
    }

    # ✅ 主返回结构不变，只新增 limit_up_table
    return {
        "topN": top_df,
        "topn": top_df,
        "full": full_sorted,
        "limit_up_table": limit_up_table,
        "meta": meta,
        "warnings": warnings,
    }


def run(df: Any, s=None) -> Dict[str, Any]:
    return run_step6_final_topn(df, s=s)


if __name__ == "__main__":
    print("Step6 FinalTopN Quant/Production Edition (Enhanced + Strength Enrich + limit_up_table) ready.")
