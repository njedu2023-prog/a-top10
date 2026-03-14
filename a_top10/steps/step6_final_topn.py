#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6 : V2 纯终排器
---------------------------------
目标：
1. 正式读取 prob_final
2. 建立唯一终排分数 final_score
3. 同时输出：
   - topN
   - full
4. 保留旧兼容字段：
   - score
   - prob
   - Probability
5. 保持主返回结构兼容：
   {
     "topN": df,
     "topn": df,
     "full": df_full,
     "meta": ...,
     "warnings": ...
   }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# =========================================================
# utils
# =========================================================
def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _coerce_df(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if _is_mapping(obj):
        for k in ("df", "data", "result", "full", "candidates", "candidate", "pool", "merged"):
            v = obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v.copy()
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v.copy()
        return pd.DataFrame()
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in cands:
        hit = lower_map.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _safe_float_series(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return s.astype("float64")


def _safe_str_series(df: pd.DataFrame, col: Optional[str], default: str = "") -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[col].astype("object").fillna(default).astype(str).str.strip()


def _clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.clip(0.0, 1.0).astype("float64")
    arr = np.asarray(s, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    return pd.Series(arr, dtype="float64")


def _stable_hash_text(x: str) -> int:
    x = str(x or "")
    return abs(hash(x)) % (10**12)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _detect_trade_date(df: pd.DataFrame, s=None) -> str:
    for c in ["trade_date", "TradeDate", "日期", "交易日期"]:
        if c in df.columns and len(df) > 0:
            v = str(df[c].iloc[0]).strip()
            if len(v) == 8 and v.isdigit():
                return v
    return ""


def _get_outputs_dir(s=None) -> Path:
    try:
        io_obj = getattr(s, "io", None)
        out = getattr(io_obj, "outputs_dir", None)
        if out:
            return Path(str(out))
    except Exception:
        pass
    return Path("outputs")


# =========================================================
# config
# =========================================================
@dataclass
class Step6Config:
    topn: int = 10
    w_prob: float = 0.70
    w_strength: float = 0.20
    w_theme: float = 0.10

    score_mode: str = "add"  # add | geo
    min_prob: float = 0.0
    min_score: Optional[float] = None

    strength_scale: float = 100.0
    theme_scale: float = 1.30

    dedup_by_ts_code: bool = True
    stable_hash_tiebreak: bool = True
    outputs_dir: str = "outputs"


def _load_config(s=None) -> Step6Config:
    cfg = Step6Config()

    try:
        io_obj = getattr(s, "io", None)
        if io_obj is not None and hasattr(io_obj, "topn"):
            cfg.topn = int(getattr(io_obj, "topn"))
    except Exception:
        pass

    try:
        step6 = getattr(s, "step6", None)
        if step6 is not None:
            for k in ["w_prob", "w_strength", "w_theme", "score_mode", "min_prob", "min_score", "strength_scale", "theme_scale", "dedup_by_ts_code", "stable_hash_tiebreak"]:
                if hasattr(step6, k):
                    setattr(cfg, k, getattr(step6, k))
    except Exception:
        pass

    try:
        io_obj = getattr(s, "io", None)
        if io_obj is not None and hasattr(io_obj, "outputs_dir"):
            cfg.outputs_dir = str(getattr(io_obj, "outputs_dir"))
    except Exception:
        pass

    cfg.topn = max(1, int(cfg.topn))
    cfg.w_prob = float(cfg.w_prob)
    cfg.w_strength = float(cfg.w_strength)
    cfg.w_theme = float(cfg.w_theme)
    cfg.score_mode = str(cfg.score_mode).lower()
    if cfg.score_mode not in ("add", "geo"):
        cfg.score_mode = "add"
    cfg.min_prob = float(cfg.min_prob)
    cfg.strength_scale = max(1e-9, float(cfg.strength_scale))
    cfg.theme_scale = max(1e-9, float(cfg.theme_scale))
    cfg.dedup_by_ts_code = bool(cfg.dedup_by_ts_code)
    cfg.stable_hash_tiebreak = bool(cfg.stable_hash_tiebreak)
    return cfg


# =========================================================
# persist pred table
# =========================================================
def _safe_write_csv(df: pd.DataFrame, path: Path) -> bool:
    try:
        _ensure_dir(path.parent)
        df.to_csv(path, index=False, encoding="utf-8")
        return True
    except Exception:
        return False


def _upsert_pred_history(path: Path, pred_df: pd.DataFrame, trade_date: str) -> bool:
    try:
        if path.exists():
            old = pd.read_csv(path, dtype=str)
        else:
            old = pd.DataFrame()

        pred_df2 = pred_df.copy().astype(str)
        if not old.empty:
            old["trade_date"] = old.get("trade_date", "").astype(str).str.strip()
            old = old[old["trade_date"] != str(trade_date)].copy()
            merged = pd.concat([old, pred_df2], ignore_index=True, sort=False)
        else:
            merged = pred_df2

        merged.to_csv(path, index=False, encoding="utf-8")
        return True
    except Exception:
        return False


# =========================================================
# core
# =========================================================
def run_step6_final_topn(df: Any, s=None) -> Dict[str, Any]:
    warnings: List[str] = []
    cfg = _load_config(s=s)

    out = _coerce_df(df)
    if out.empty:
        empty = pd.DataFrame()
        return {"topN": empty, "topn": empty, "full": empty, "meta": {"empty": True}, "warnings": warnings}

    # ---------- identify columns ----------
    ts_col = _first_existing_col(out, ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"])
    name_col = _first_existing_col(out, ["name", "stock_name", "名称", "股票简称", "证券名称"])
    board_col = _first_existing_col(out, ["board", "industry", "板块", "行业", "concept", "theme", "题材"])

    prob_final_col = _first_existing_col(out, ["prob_final"])
    prob_rule_col = _first_existing_col(out, ["prob_rule"])
    prob_ml_col = _first_existing_col(out, ["prob_ml"])

    # 兼容旧字段：如果还没有 prob_final，则退回 Probability
    if prob_final_col is None:
        prob_final_col = _first_existing_col(out, ["Probability", "probability", "prob", "_prob", "预测概率", "概率"])
        if prob_final_col is not None:
            warnings.append("step6 fallback: prob_final missing, fallback to old Probability field.")

    strength_col = _first_existing_col(out, ["StrengthScore", "strengthscore", "strength", "_strength", "强度得分", "强度"])
    theme_col = _first_existing_col(out, ["ThemeBoost", "themeboost", "theme_boost", "theme", "_theme", "题材加成", "热度"])

    # ---------- normalize ----------
    ts_series = _safe_str_series(out, ts_col, "")
    name_series = _safe_str_series(out, name_col, "")
    board_series = _safe_str_series(out, board_col, "")

    prob_final = _clip01(_safe_float_series(out, prob_final_col, 0.0))
    prob_rule = _clip01(_safe_float_series(out, prob_rule_col, 0.0)) if prob_rule_col else pd.Series([0.0] * len(out), index=out.index)
    prob_ml = _safe_float_series(out, prob_ml_col, np.nan) if prob_ml_col else pd.Series([np.nan] * len(out), index=out.index)

    strength = _safe_float_series(out, strength_col, 0.0).clip(lower=0.0)
    theme = _safe_float_series(out, theme_col, 0.0).clip(lower=0.0)

    strength01 = _clip01(strength / cfg.strength_scale)
    theme01 = _clip01(theme / cfg.theme_scale)

    # ---------- scoring ----------
    w_sum = max(1e-12, cfg.w_prob + cfg.w_strength + cfg.w_theme)
    w_prob = cfg.w_prob / w_sum
    w_strength = cfg.w_strength / w_sum
    w_theme = cfg.w_theme / w_sum

    if cfg.score_mode == "geo":
        final_score = (
            np.power(np.maximum(prob_final, 1e-12), w_prob)
            * np.power(np.maximum(strength01, 1e-12), w_strength)
            * np.power(np.maximum(theme01, 1e-12), w_theme)
        )
        final_score = pd.Series(final_score, index=out.index, dtype="float64")
    else:
        final_score = (
            w_prob * prob_final
            + w_strength * strength01
            + w_theme * theme01
        ).astype("float64")

    # ---------- filters ----------
    out2 = out.copy()
    out2["prob_rule"] = prob_rule.astype("float64")
    out2["prob_ml"] = prob_ml.astype("float64")
    out2["prob_final"] = prob_final.astype("float64")
    out2["Probability"] = out2["prob_final"].astype("float64")
    out2["StrengthScore"] = strength.astype("float64")
    out2["ThemeBoost"] = theme.astype("float64")
    out2["strength01"] = strength01.astype("float64")
    out2["theme01"] = theme01.astype("float64")
    out2["final_score"] = final_score.astype("float64")
    out2["score"] = out2["final_score"].astype("float64")   # 兼容旧 score
    out2["prob"] = out2["prob_final"].astype("float64")     # 兼容旧 prob
    out2["ts_code"] = ts_series
    out2["name"] = name_series
    out2["board"] = board_series

    if cfg.min_prob > 0:
        before = len(out2)
        out2 = out2[out2["prob_final"] >= cfg.min_prob].copy()
        if len(out2) < before:
            warnings.append(f"filtered by min_prob={cfg.min_prob}: {before}->{len(out2)}")

    if cfg.min_score is not None:
        before = len(out2)
        out2 = out2[out2["final_score"] >= float(cfg.min_score)].copy()
        if len(out2) < before:
            warnings.append(f"filtered by min_score={cfg.min_score}: {before}->{len(out2)}")

    if out2.empty:
        empty = pd.DataFrame()
        return {"topN": empty, "topn": empty, "full": empty, "meta": {"filtered_all": True}, "warnings": warnings}

    # ---------- dedup ----------
    if cfg.dedup_by_ts_code and "ts_code" in out2.columns:
        out2 = out2.sort_values(
            ["final_score", "prob_final", "StrengthScore", "ts_code"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).drop_duplicates(subset=["ts_code"], keep="first").copy()

    # ---------- stable tie-break ----------
    if cfg.stable_hash_tiebreak:
        out2["_tiebreak_hash"] = [
            _stable_hash_text(f"{a}||{b}||{c}")
            for a, b, c in zip(out2["ts_code"], out2["name"], out2["board"])
        ]
    else:
        out2["_tiebreak_hash"] = 0

    # ---------- full ranking ----------
    full_sorted = out2.sort_values(
        ["final_score", "prob_final", "StrengthScore", "_tiebreak_hash"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    full_sorted["rank"] = np.arange(1, len(full_sorted) + 1)

    prefer_cols = [
        "rank",
        "ts_code",
        "name",
        "board",
        "StrengthScore",
        "ThemeBoost",
        "prob_rule",
        "prob_ml",
        "prob_final",
        "Probability",
        "final_score",
        "score",
        "prob",
        "strength01",
        "theme01",
    ]
    exist = [c for c in prefer_cols if c in full_sorted.columns]
    others = [c for c in full_sorted.columns if c not in exist and c != "_tiebreak_hash"]
    full_sorted = full_sorted[exist + others]

    top_df = full_sorted.head(cfg.topn).copy()

    # ---------- persist top10 pred table ----------
    pred_meta: Dict[str, Any] = {
        "written": False,
        "daily_path": "",
        "history_path": "",
        "trade_date": "",
    }
    try:
        trade_date = _detect_trade_date(full_sorted, s=s)
        if trade_date:
            outputs_dir = _get_outputs_dir(s)
            learning_dir = outputs_dir / "learning"
            _ensure_dir(learning_dir)

            daily_path = learning_dir / f"pred_top10_{trade_date}.csv"
            history_path = learning_dir / "pred_top10_history.csv"

            pred_df = top_df.copy()
            pred_df.insert(0, "trade_date", trade_date)

            ok_daily = _safe_write_csv(pred_df, daily_path)
            ok_hist = _upsert_pred_history(history_path, pred_df, trade_date)

            pred_meta.update({
                "written": bool(ok_daily and ok_hist),
                "daily_path": str(daily_path),
                "history_path": str(history_path),
                "trade_date": trade_date,
            })
    except Exception as e:
        warnings.append(f"pred_table_exception: {e}")

    meta = {
        "rows_in": int(len(out)),
        "rows_out": int(len(full_sorted)),
        "topn": int(cfg.topn),
        "score_mode": cfg.score_mode,
        "weights": {
            "w_prob": float(w_prob),
            "w_strength": float(w_strength),
            "w_theme": float(w_theme),
        },
        "columns_used": {
            "ts_col": ts_col,
            "name_col": name_col,
            "board_col": board_col,
            "prob_final_col": prob_final_col,
            "strength_col": strength_col,
            "theme_col": theme_col,
        },
        "pred_table": pred_meta,
        "v2_semantics": {
            "main_probability_field": "prob_final",
            "compat_probability_field": "Probability",
            "main_score_field": "final_score",
        },
    }

    return {
        "topN": top_df,
        "topn": top_df,
        "full": full_sorted,
        "meta": meta,
        "warnings": warnings,
    }


def run(df: Any, s=None) -> Dict[str, Any]:
    return run_step6_final_topn(df, s=s)
