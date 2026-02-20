#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : 概率模型推断（ML核心层） + 可训练闭环（LR + LightGBM）

✅ 关键增强（为了解决 feature_history 全 0）：
- 写 feature_history.csv 前，若 StrengthScore/turnover_rate/seal_amount/open_times 缺失或全 0，
  则自动尝试从 outputs/step3_strength_{trade_date}.csv 回补（按 ts_code merge）。
"""

from __future__ import annotations

import os
import json
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from a_top10.config import Settings

FEATURES = [
    "StrengthScore",
    "ThemeBoost",
    "seal_amount",
    "open_times",
    "turnover_rate",
]


# -------------------------
# Utils
# -------------------------
def _ensure_df(x) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _first_existing_col(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"])


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    ts_col = _get_ts_code_col(df)
    if ts_col and ts_col != "ts_code":
        df["ts_code"] = df[ts_col].astype(str)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.strip()
    return df


def _safe_log1p_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    return np.log1p(x)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _guess_trade_date(df: pd.DataFrame) -> str:
    df = _ensure_df(df)
    for c in ["trade_date", "TradeDate", "日期", "交易日期"]:
        if c in df.columns and len(df) > 0:
            v = str(df[c].iloc[0]).strip()
            if re.match(r"^\d{8}$", v):
                return v

    env_td = os.getenv("TRADE_DATE", "").strip()
    if re.match(r"^\d{8}$", env_td):
        return env_td

    return pd.Timestamp.today().strftime("%Y%m%d")


def _stable_jitter_from_ts(ts: str) -> float:
    s = str(ts or "").strip()
    if not s:
        return 0.0
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    v = int(h, 16) / float(16**8)
    return (v - 0.5) * 2e-4


def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_sampling_state(outputs_dir: Path) -> Dict[str, Any]:
    p = outputs_dir / "learning" / "sampling_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_target_rows_per_day(s=None) -> int:
    outputs_dir = Path(getattr(getattr(s, "io", None), "outputs_dir", "outputs")) if s is not None else Path("outputs")
    st = _read_sampling_state(outputs_dir)
    try:
        v = int(st.get("target_rows_per_day", 200))
    except Exception:
        v = 200
    return int(max(50, min(2000, v)))


def _read_step3_strength_file(outputs_dir: Path, trade_date: str) -> pd.DataFrame:
    p = outputs_dir / f"step3_strength_{trade_date}.csv"
    if not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc)
        except Exception:
            continue
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        return pd.DataFrame()


def _backfill_features_from_step3(raw_input_df: pd.DataFrame, trade_date: str, outputs_dir: Path) -> pd.DataFrame:
    """
    ✅ 硬兜底：从 outputs/step3_strength_{trade_date}.csv 回补关键特征
    """
    df = _normalize_id_columns(_ensure_df(raw_input_df)).copy()
    if df.empty or "ts_code" not in df.columns:
        return df

    step3 = _read_step3_strength_file(outputs_dir, trade_date)
    if step3 is None or step3.empty:
        return df

    step3 = _normalize_id_columns(step3)
    if "ts_code" not in step3.columns:
        return df

    need_cols = ["StrengthScore", "turnover_rate", "seal_amount", "open_times"]
    for c in need_cols:
        if c not in step3.columns:
            step3[c] = "0"

    step3_small = step3[["ts_code"] + need_cols].copy()
    for c in need_cols:
        step3_small[c] = pd.to_numeric(step3_small[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    merged = df.merge(step3_small, on="ts_code", how="left", suffixes=("", "_s3"))

    for c in need_cols:
        # 若原列缺失或全 0，则用 step3 回补
        if c not in merged.columns:
            merged[c] = 0.0
        x = pd.to_numeric(merged[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = pd.to_numeric(merged.get(f"{c}_s3", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if float((x != 0.0).mean()) < 0.05 and float((y != 0.0).mean()) > 0.05:
            merged[c] = y
        else:
            # 仅对空值回补
            merged[c] = x.where(x != 0.0, y)

        if f"{c}_s3" in merged.columns:
            merged = merged.drop(columns=[f"{c}_s3"])

    return merged


def _write_feature_history(
    raw_input_df: pd.DataFrame,
    out_df: pd.DataFrame,
    trade_date: str,
    s=None,
) -> Dict[str, Any]:
    try:
        outputs_dir = Path(getattr(getattr(s, "io", None), "outputs_dir", "outputs")) if s is not None else Path("outputs")

        raw_input_df = _normalize_id_columns(_ensure_df(raw_input_df))
        out_df = _ensure_df(out_df)

        if raw_input_df.empty:
            return {"ok": False, "reason": "empty raw_input"}
        if "ts_code" not in raw_input_df.columns:
            return {"ok": False, "reason": "missing ts_code in raw_input"}

        # ✅ 回补关键特征（解决 feature_history 关键列全 0）
        raw_input_df = _backfill_features_from_step3(raw_input_df, trade_date=trade_date, outputs_dir=outputs_dir)

        target_n = _get_target_rows_per_day(s=s)

        tmp = pd.DataFrame()
        tmp["ts_code"] = raw_input_df["ts_code"].astype(str).str.strip()

        name_col = None
        for c in ["name", "stock_name", "名称", "股票简称", "证券名称"]:
            if c in raw_input_df.columns:
                name_col = c
                break
        tmp["name"] = raw_input_df[name_col].astype(str) if name_col else ""

        feat = _ensure_features(raw_input_df)
        for c in FEATURES:
            tmp[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        tmp["Probability"] = pd.to_numeric(out_df.get("Probability", np.nan), errors="coerce")
        tmp["_prob_src"] = out_df.get("_prob_src", "").astype(str).fillna("") if "_prob_src" in out_df.columns else ""
        tmp["_prob_warn"] = out_df.get("_prob_warn", "").astype(str).fillna("") if "_prob_warn" in out_df.columns else ""

        tmp["run_time_utc"] = _utc_now_iso()
        tmp["trade_date"] = str(trade_date).strip()

        tmp["_prob_f"] = pd.to_numeric(tmp["Probability"], errors="coerce").fillna(0.0)
        tmp = tmp.sort_values("_prob_f", ascending=False).drop(columns=["_prob_f"])
        if target_n > 0:
            tmp = tmp.head(int(target_n))

        base = outputs_dir / "learning"
        _ensure_dir(base)
        fp = base / "feature_history.csv"

        if fp.exists():
            old = pd.read_csv(fp, dtype=str, encoding="utf-8")
            merged = pd.concat([old, tmp.astype(str)], ignore_index=True, sort=False)
        else:
            merged = tmp.astype(str)

        merged["trade_date"] = merged.get("trade_date", "").astype(str).str.strip()
        merged["ts_code"] = merged.get("ts_code", "").astype(str).str.strip()
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

        try:
            merged["_prob_f"] = pd.to_numeric(merged.get("Probability", 0), errors="coerce").fillna(0.0)
            merged = merged.sort_values(["trade_date", "_prob_f"], ascending=[True, False]).drop(columns=["_prob_f"])
        except Exception:
            merged = merged.sort_values(["trade_date"], ascending=[True])

        merged.to_csv(fp, index=False, encoding="utf-8")
        return {
            "ok": True,
            "path": str(fp),
            "rows": int(len(merged)),
            "trade_date": str(trade_date),
            "wrote_rows_today": int(len(tmp)),
            "target_rows_per_day": int(target_n),
        }
    except Exception as e:
        return {"ok": False, "reason": f"exception: {e}"}


# -------------------------
# Model IO
# -------------------------
@dataclass
class Step5ModelPaths:
    base: Path
    lr_path: Path
    lgbm_path: Path


def _get_model_paths(s=None) -> Step5ModelPaths:
    base = None
    if s is not None and hasattr(s, "data_repo"):
        dr = s.data_repo
        for attr in ["models_dir", "model_dir"]:
            if hasattr(dr, attr):
                try:
                    base = Path(getattr(dr, attr))
                    break
                except Exception:
                    base = None
        if base is None and hasattr(dr, "root"):
            try:
                base = Path(dr.root) / "models"
            except Exception:
                base = None

    if base is None:
        base = Path("models")

    base.mkdir(parents=True, exist_ok=True)
    return Step5ModelPaths(
        base=base,
        lr_path=base / "step5_lr.joblib",
        lgbm_path=base / "step5_lgbm.joblib",
    )


def _load_joblib(path: Path):
    if joblib is None:
        return None
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _save_joblib(obj, path: Path) -> None:
    if joblib is None:
        return
    try:
        joblib.dump(obj, path)
    except Exception:
        pass


def load_lr(s=None):
    paths = _get_model_paths(s=s)
    return _load_joblib(paths.lr_path)


def load_lgbm(s=None):
    paths = _get_model_paths(s=s)
    return _load_joblib(paths.lgbm_path)


# -------------------------
# Train（保持原接口）
# -------------------------
def _load_step4_theme_history(s, lookback: int, theme_file_name: str) -> pd.DataFrame:
    outputs_dir = Path(getattr(s.io, "outputs_dir", "outputs"))
    fp = outputs_dir / "learning" / theme_file_name
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding="gbk")
        except Exception:
            return pd.DataFrame()


def _build_X_y_from_theme_history(s, lookback: int, theme_file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    hist = _load_step4_theme_history(s, lookback=lookback, theme_file_name=theme_file_name)
    hist = _normalize_id_columns(hist)
    if hist.empty or "trade_date" not in hist.columns or "ts_code" not in hist.columns:
        return np.zeros((0, len(FEATURES))), np.zeros((0,))

    dates = sorted(hist["trade_date"].astype(str).unique())
    rows = []
    y = []
    for i, d in enumerate(dates[:-1]):
        next_d = dates[i + 1]
        df_day = hist[hist["trade_date"].astype(str) == str(d)].copy()
        if df_day.empty:
            continue

        try:
            df_limit = s.data_repo.read_limit_list(next_d)
        except Exception:
            df_limit = pd.DataFrame()

        lim = set()
        if df_limit is not None and not df_limit.empty:
            df_limit = _normalize_id_columns(df_limit)
            if "ts_code" in df_limit.columns:
                lim = set(df_limit["ts_code"].astype(str).tolist())

        feat = _ensure_features(df_day)
        for j in range(len(df_day)):
            rows.append(feat[FEATURES].iloc[j].astype(float).values)
            code = str(df_day["ts_code"].iloc[j]).strip()
            y.append(1 if code in lim else 0)

    if not rows:
        return np.zeros((0, len(FEATURES))), np.zeros((0,))
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=int)


def train_step5_lr(s, lookback: int = 90, theme_file_name: str = "step4_theme.csv") -> Dict[str, Any]:
    X, y = _build_X_y_from_theme_history(s, lookback=lookback, theme_file_name=theme_file_name)
    if X.shape[0] < 50 or len(np.unique(y)) < 2:
        return {"ok": False, "reason": "not enough samples or single class", "n": int(X.shape[0]), "pos": int(y.sum())}

    model = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True)),
                            ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))])
    model.fit(X, y)

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.lr_path)
    return {"ok": True, "path": str(paths.lr_path), "n": int(X.shape[0]), "pos": int(y.sum())}


def train_step5_lgbm(s, lookback: int = 120, theme_file_name: str = "step4_theme.csv") -> Dict[str, Any]:
    if LGBMClassifier is None:
        return {"ok": False, "reason": "lightgbm not installed"}

    X, y = _build_X_y_from_theme_history(s, lookback=lookback, theme_file_name=theme_file_name)
    if X.shape[0] < 80 or len(np.unique(y)) < 2:
        return {"ok": False, "reason": "not enough samples or single class", "n": int(X.shape[0]), "pos": int(y.sum())}

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    model.fit(X, y)

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.lgbm_path)
    return {"ok": True, "path": str(paths.lgbm_path), "n": int(X.shape[0]), "pos": int(y.sum())}


def train_step5_models(s, lookback: int = 120, theme_file_name: str = "step4_theme.csv") -> Dict[str, Any]:
    res_lr = train_step5_lr(s, lookback=max(60, int(lookback * 0.75)), theme_file_name=theme_file_name)
    res_lgbm = train_step5_lgbm(s, lookback=lookback, theme_file_name=theme_file_name)
    ok_any = bool(res_lr.get("ok")) or bool(res_lgbm.get("ok"))
    return {"ok": ok_any, "lr": res_lr, "lgbm": res_lgbm}


# -------------------------
# Inference
# -------------------------
def run_step5(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    raw_input = _ensure_df(theme_df)
    if raw_input.empty:
        out = pd.DataFrame()
        out["Probability"] = pd.Series(dtype="float64")
        out["_prob_src"] = pd.Series(dtype="object")
        return out

    out = raw_input.copy()

    ts_norm = _normalize_id_columns(raw_input)
    ts_series = ts_norm["ts_code"].astype(str).str.strip() if "ts_code" in ts_norm.columns else pd.Series([""] * len(out))

    feat = _ensure_features(raw_input)
    X = feat[FEATURES].astype(float).values

    def _pseudo_proba() -> np.ndarray:
        strength = np.clip(feat["StrengthScore"].astype(float).values / 100.0, 0.0, 1.5)
        theme = np.clip(feat["ThemeBoost"].astype(float).values, 0.0, 2.0)

        seal = _safe_log1p_pos(feat["seal_amount"].astype(float).values)
        seal = np.clip(seal / 16.0, 0.0, 1.5)

        opens = np.clip(feat["open_times"].astype(float).values, 0.0, 20.0)
        turnover = np.clip(feat["turnover_rate"].astype(float).values, 0.0, 50.0) / 50.0

        z = 1.20 * strength + 1.10 * theme + 0.90 * seal + 0.35 * turnover - 0.60 * (opens / 10.0)
        return _sigmoid(z)

    def _post_process(proba: np.ndarray, src: str) -> pd.DataFrame:
        proba = np.asarray(proba, dtype=float)
        proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba + np.array([_stable_jitter_from_ts(t) for t in ts_series.values], dtype=float)

        try:
            if np.nanstd(proba) < 1e-8:
                p2 = _pseudo_proba()
                proba = 0.80 * proba + 0.20 * p2
                src = f"{src}+tie"
        except Exception:
            pass

        proba = np.clip(proba, 0.0, 1.0)

        out2 = out.copy()
        out2["Probability"] = proba
        out2["_prob_src"] = src

        td = _guess_trade_date(raw_input)
        _write_feature_history(raw_input_df=raw_input, out_df=out2, trade_date=td, s=s)
        return out2.sort_values("Probability", ascending=False)

    m_lgbm = load_lgbm(s=s)
    if m_lgbm is not None:
        try:
            proba = m_lgbm.predict_proba(X)[:, 1]
            return _post_process(proba, "lgbm")
        except Exception:
            pass

    m_lr = load_lr(s=s)
    if m_lr is not None:
        try:
            proba = m_lr.predict_proba(X)[:, 1]
            return _post_process(proba, "lr")
        except Exception:
            pass

    return _post_process(_pseudo_proba(), "pseudo")


def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(df, s=s)


if __name__ == "__main__":
    print("Step5 Probability model loaded.")
