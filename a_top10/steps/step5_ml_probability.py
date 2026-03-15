#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : ML Probability — Top10 V3

定位：
- Step5 不再是 V2 的“prob_ml / prob_final 兼容层”，而是 V3 的主概率引擎。
- 明确输出：
    - prob_lr
    - prob_lgbm
    - prob_rule
    - Probability
    - _prob_src
- 其中：
    - Probability = 最终主排序概率轴
    - _prob_src = 最终概率来源，必须可追踪

兼容目标：
- 保留训练接口：
    - train_step5_lr
    - train_step5_lgbm
    - train_step5_models
- 保留推理入口：
    - run_step5(theme_df, s=None)
    - run(theme_df, s=None)

V3 原则：
- Step5 内部计算允许对缺失特征做数值兜底，但不得污染上游契约字段
- 若 ML 模型不可用，则必须显式回退到 prob_rule，并写明 _prob_src
- 若 Probability 无法得到，则样本不应被当作正常概率样本消费
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


# =========================================================
# Config
# =========================================================

FEATURES = [
    "StrengthScore",
    "ThemeBoost",
    "seal_amount",
    "open_times",
    "turnover_rate",
]

VALID_RUN_MODES = {"replay", "train", "auto_daily"}


# =========================================================
# Basic utils
# =========================================================

def _ensure_df(x: Any) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _first_existing_col(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in cands:
        hit = lower_map.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"])


def _get_name_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["name", "stock_name", "名称", "股票简称", "证券名称"])


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_df(df)
    ts_col = _get_ts_code_col(df)
    if ts_col and ts_col != "ts_code":
        df["ts_code"] = df[ts_col].astype(str)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.strip()
    return df


def _to_numeric_nullable(sr: pd.Series) -> pd.Series:
    return pd.to_numeric(sr, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.astype("float64").clip(0.0, 1.0)
    arr = np.asarray(s, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    return pd.Series(arr, dtype="float64")


def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _get_outputs_dir(s=None) -> Path:
    try:
        io_obj = getattr(s, "io", None)
        out = getattr(io_obj, "outputs_dir", None)
        if out:
            return Path(str(out))
    except Exception:
        pass
    return Path("outputs")


def _get_ml_cfg(s=None) -> Dict[str, Any]:
    cfg = {
        "model": "auto",                  # auto | logistic | lightgbm
        "enable_rule": True,
        "enable_ml": True,
        "fusion_mode": "ml_first",       # ml_first | weighted
        "fallback_to_rule": True,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "rule_weight": 0.30,
        "ml_weight": 0.70,
    }
    try:
        ml = getattr(s, "ml", None)
        if ml is None:
            return cfg
        for k in list(cfg.keys()):
            if hasattr(ml, k):
                cfg[k] = getattr(ml, k)
    except Exception:
        pass

    cfg["model"] = str(cfg.get("model", "auto")).lower()
    cfg["fusion_mode"] = str(cfg.get("fusion_mode", "ml_first")).lower()
    cfg["enable_rule"] = bool(cfg.get("enable_rule", True))
    cfg["enable_ml"] = bool(cfg.get("enable_ml", True))
    cfg["fallback_to_rule"] = bool(cfg.get("fallback_to_rule", True))
    cfg["clip_min"] = float(cfg.get("clip_min", 0.0))
    cfg["clip_max"] = float(cfg.get("clip_max", 1.0))
    cfg["rule_weight"] = float(cfg.get("rule_weight", 0.30))
    cfg["ml_weight"] = float(cfg.get("ml_weight", 0.70))
    return cfg


# =========================================================
# Run mode / training policy
# =========================================================

@dataclass
class Step5TrainingPolicy:
    run_mode: str
    allow_model_update_raw: str
    allow_model_update: bool
    min_level1_samples: int
    min_train_samples: int
    min_stable_samples: int
    min_positive_samples: int
    min_feature_coverage: float


def _resolve_run_mode() -> str:
    mode = os.getenv("A_TOP10_RUN_MODE", os.getenv("TOP10_RUN_MODE", "auto_daily")).strip().lower()
    if mode not in VALID_RUN_MODES:
        return "auto_daily"
    return mode


def _resolve_allow_model_update(run_mode: str) -> Tuple[str, bool]:
    raw = os.getenv(
        "A_TOP10_ALLOW_MODEL_UPDATE",
        os.getenv("TOP10_ALLOW_MODEL_UPDATE", "auto"),
    ).strip().lower()

    if raw in {"1", "true", "yes", "on"}:
        return raw, True
    if raw in {"0", "false", "no", "off"}:
        return raw, False

    if run_mode == "replay":
        return "auto", False
    if run_mode == "train":
        return "auto", True
    return "auto", True


def _policy_value_from_settings(s: Any, *names: str, default: Any) -> Any:
    for obj_name in ["training", "ml"]:
        try:
            obj = getattr(s, obj_name, None)
            if obj is None:
                continue
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    if v is not None:
                        return v
        except Exception:
            pass
    return default


def _get_training_policy(s=None) -> Step5TrainingPolicy:
    run_mode = _resolve_run_mode()
    allow_raw, allow_update = _resolve_allow_model_update(run_mode)

    min_level1_samples = int(_policy_value_from_settings(s, "min_level1_samples", default=80))
    min_train_samples = int(_policy_value_from_settings(s, "min_train_samples", default=120))
    min_stable_samples = int(_policy_value_from_settings(s, "min_stable_samples", default=150))
    min_positive_samples = int(_policy_value_from_settings(s, "min_positive_samples", default=12))
    min_feature_coverage = float(_policy_value_from_settings(s, "min_feature_coverage", default=0.85))

    return Step5TrainingPolicy(
        run_mode=run_mode,
        allow_model_update_raw=allow_raw,
        allow_model_update=allow_update,
        min_level1_samples=min_level1_samples,
        min_train_samples=min_train_samples,
        min_stable_samples=min_stable_samples,
        min_positive_samples=min_positive_samples,
        min_feature_coverage=min_feature_coverage,
    )


def _classify_training_level(n_samples: int, policy: Step5TrainingPolicy) -> str:
    if n_samples < policy.min_level1_samples:
        return "below_level1"
    if n_samples < policy.min_train_samples:
        return "level1"
    if n_samples < policy.min_stable_samples:
        return "level2"
    return "level3"


def _can_formal_update(level: str, policy: Step5TrainingPolicy) -> bool:
    if not policy.allow_model_update:
        return False
    return level in {"level2", "level3"}


# =========================================================
# Step3 feature backfill
# =========================================================

def _backfill_features_from_step3(raw_df: pd.DataFrame, trade_date: str, outputs_dir: Path) -> pd.DataFrame:
    """
    允许从 Step3 产物回填核心特征，但只在当前 df 缺列或几乎全空/全零时回填。
    """
    raw_df = _normalize_id_columns(raw_df)
    if raw_df.empty or "ts_code" not in raw_df.columns:
        return raw_df

    need_cols = ["StrengthScore", "seal_amount", "open_times", "turnover_rate"]
    missing = []

    for c in need_cols:
        if c not in raw_df.columns:
            missing.append(c)
            continue

        vals = _to_numeric_nullable(raw_df[c])
        nonnull_ratio = float(vals.notna().mean()) if len(vals) else 0.0
        nonzero_ratio = float((vals.fillna(0.0) != 0.0).mean()) if len(vals) else 0.0

        if nonnull_ratio < 0.01 or nonzero_ratio < 0.01:
            missing.append(c)

    if not missing:
        return raw_df

    candidates = [
        outputs_dir / f"step3_strength_{trade_date}.csv",
        outputs_dir / "step3_strength.csv",
        Path("outputs") / f"step3_strength_{trade_date}.csv",
        Path("outputs") / "step3_strength.csv",
    ]
    src = None
    for p in candidates:
        if p.exists():
            src = p
            break
    if src is None:
        return raw_df

    try:
        step3_df = pd.read_csv(src)
    except Exception:
        return raw_df

    step3_df = _normalize_id_columns(step3_df)
    if step3_df.empty or "ts_code" not in step3_df.columns:
        return raw_df

    use_cols = ["ts_code"] + [c for c in need_cols if c in step3_df.columns]
    if len(use_cols) <= 1:
        return raw_df

    merged = raw_df.merge(step3_df[use_cols], on="ts_code", how="left", suffixes=("", "_s3"))

    for c in need_cols:
        s3 = f"{c}_s3"
        if c in merged.columns and s3 in merged.columns:
            cur = _to_numeric_nullable(merged[c])
            ext = _to_numeric_nullable(merged[s3])

            cur_bad = cur.isna() | (cur.fillna(0.0) == 0.0)
            merged[c] = cur.where(~cur_bad, ext)
            merged.drop(columns=[s3], inplace=True, errors="ignore")
        elif c not in merged.columns and s3 in merged.columns:
            merged[c] = _to_numeric_nullable(merged[s3])
            merged.drop(columns=[s3], inplace=True, errors="ignore")

    return merged


# =========================================================
# Inference feature prep
# =========================================================

def _ensure_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅为 Step5 内部推理准备数值特征。
    不对外伪造契约字段。
    """
    df = _ensure_df(df)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    feat = df.copy()
    for c in FEATURES:
        feat[c] = _to_numeric_nullable(feat[c]).fillna(0.0)

    return feat


# =========================================================
# Model IO
# =========================================================

@dataclass
class Step5ModelPaths:
    base: Path
    lr_path: Path
    lgbm_path: Path


def _get_model_paths(s=None) -> Step5ModelPaths:
    base = Path("models")
    try:
        dr = getattr(s, "data_repo", None)
        if dr is not None:
            for attr in ["models_dir", "model_dir"]:
                if hasattr(dr, attr):
                    cand = getattr(dr, attr)
                    if cand:
                        base = Path(str(cand))
                        break
    except Exception:
        pass

    _ensure_dir(base)
    return Step5ModelPaths(
        base=base,
        lr_path=base / "step5_lr.joblib",
        lgbm_path=base / "step5_lgbm.joblib",
    )


def _load_joblib(path: Path):
    if joblib is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _save_joblib(obj: Any, path: Path) -> None:
    if joblib is None:
        return
    try:
        joblib.dump(obj, path)
    except Exception:
        pass


def load_lr(s=None):
    return _load_joblib(_get_model_paths(s).lr_path)


def load_lgbm(s=None):
    return _load_joblib(_get_model_paths(s).lgbm_path)


# =========================================================
# Training dataset
# =========================================================

def _load_feature_history(s) -> pd.DataFrame:
    outputs_dir = _get_outputs_dir(s)
    fp = outputs_dir / "learning" / "feature_history.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding="gbk")
        except Exception:
            return pd.DataFrame()


def _sample_feature_coverage(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    coverages = []
    for c in FEATURES:
        if c not in df.columns:
            coverages.append(0.0)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        coverages.append(float(s.notna().mean()))
    return float(np.mean(coverages)) if coverages else 0.0


def _build_X_y_from_feature_history(
    s,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    hist = _load_feature_history(s)
    hist = _normalize_id_columns(hist)

    required = {"trade_date", "ts_code", "y_limit_hit", "is_sample_mature"}
    if hist.empty or not required.issubset(set(hist.columns)):
        return np.zeros((0, len(FEATURES))), np.zeros((0,)), {
            "mature_samples": 0,
            "positive_samples": 0,
            "feature_coverage": 0.0,
            "reason": "feature_history_missing_or_unlabeled",
        }

    hist["trade_date"] = hist["trade_date"].astype(str).str.strip()
    dates = sorted(hist["trade_date"].unique())
    if lookback > 0:
        dates = dates[-lookback:]

    rows = []
    y = []
    coverage_list = []

    for d in dates:
        df_day = hist[hist["trade_date"] == str(d)].copy()
        if df_day.empty:
            continue

        mature = pd.to_numeric(df_day.get("is_sample_mature"), errors="coerce").fillna(0.0)
        label = pd.to_numeric(df_day.get("y_limit_hit"), errors="coerce")

        df_day = df_day[(mature > 0.5) & label.notna()].copy()
        if df_day.empty:
            continue

        coverage_list.append(_sample_feature_coverage(df_day))

        feat = _ensure_inference_input(df_day)
        yy = pd.to_numeric(df_day["y_limit_hit"], errors="coerce").fillna(0.0).astype(int)

        for i in range(len(df_day)):
            rows.append(feat[FEATURES].iloc[i].astype(float).values)
            y.append(int(yy.iloc[i]))

    if not rows:
        return np.zeros((0, len(FEATURES))), np.zeros((0,)), {
            "mature_samples": 0,
            "positive_samples": 0,
            "feature_coverage": 0.0,
            "reason": "no_mature_labeled_rows",
        }

    X = np.asarray(rows, dtype=float)
    yy = np.asarray(y, dtype=int)
    meta = {
        "mature_samples": int(X.shape[0]),
        "positive_samples": int(yy.sum()),
        "feature_coverage": float(np.mean(coverage_list)) if coverage_list else 0.0,
        "reason": "ok",
    }
    return X, yy, meta


def _training_gate_summary(
    n_samples: int,
    pos_samples: int,
    feature_coverage: float,
    policy: Step5TrainingPolicy,
) -> Dict[str, Any]:
    level = _classify_training_level(n_samples, policy)
    formal_update_allowed = _can_formal_update(level, policy)

    summary = {
        "run_mode": policy.run_mode,
        "allow_model_update_raw": policy.allow_model_update_raw,
        "allow_model_update": bool(policy.allow_model_update),
        "level": level,
        "mature_samples": int(n_samples),
        "positive_samples": int(pos_samples),
        "feature_coverage": float(feature_coverage),
        "formal_update_allowed": bool(formal_update_allowed),
        "trial_training_only": level == "level1",
    }

    if level == "below_level1":
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "below_level1_min_samples"})
        return summary

    if pos_samples < policy.min_positive_samples:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "insufficient_positive_samples"})
        return summary

    if feature_coverage < policy.min_feature_coverage:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "insufficient_feature_coverage"})
        return summary

    summary.update({"ok": True, "trained": True, "updated": False, "reason": "ready_for_training"})
    return summary


def train_step5_lr(s, lookback: int = 120, theme_file_name: str = "step4_theme.csv") -> Dict[str, Any]:
    """
    保留旧签名兼容，但 V3 正式训练不再依赖 step4_theme.csv，
    而是依赖已成熟、已打标签的 feature_history.csv。
    """
    policy = _get_training_policy(s)
    X, y, meta = _build_X_y_from_feature_history(s, lookback=lookback)

    n_samples = int(meta.get("mature_samples", 0))
    pos_samples = int(meta.get("positive_samples", 0))
    feature_coverage = float(meta.get("feature_coverage", 0.0))

    summary = _training_gate_summary(
        n_samples=n_samples,
        pos_samples=pos_samples,
        feature_coverage=feature_coverage,
        policy=policy,
    )
    summary["data_reason"] = meta.get("reason", "")

    if not summary.get("ok"):
        return summary

    if len(np.unique(y)) < 2:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "single_class_labels"})
        return summary

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    summary["trained"] = True

    level = str(summary.get("level", ""))
    if level == "level1":
        summary["updated"] = False
        summary["reason"] = "level1_trial_training_no_formal_update"
        return summary

    if not summary.get("formal_update_allowed", False):
        summary["updated"] = False
        summary["reason"] = "model_update_not_allowed_by_run_mode"
        return summary

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.lr_path)
    summary["updated"] = True
    summary["path"] = str(paths.lr_path)
    summary["reason"] = "formal_model_updated"
    return summary


def train_step5_lgbm(s, lookback: int = 150, theme_file_name: str = "step4_theme.csv") -> Dict[str, Any]:
    policy = _get_training_policy(s)

    if LGBMClassifier is None:
        return {
            "ok": False,
            "trained": False,
            "updated": False,
            "run_mode": policy.run_mode,
            "reason": "lightgbm_not_installed",
        }

    X, y, meta = _build_X_y_from_feature_history(s, lookback=lookback)
    n_samples = int(meta.get("mature_samples", 0))
    pos_samples = int(meta.get("positive_samples", 0))
    feature_coverage = float(meta.get("feature_coverage", 0.0))

    summary = _training_gate_summary(
        n_samples=n_samples,
        pos_samples=pos_samples,
        feature_coverage=feature_coverage,
        policy=policy,
    )
    summary["data_reason"] = meta.get("reason", "")

    if not summary.get("ok"):
        return summary

    if len(np.unique(y)) < 2:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "single_class_labels"})
        return summary

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    model.fit(X, y)
    summary["trained"] = True

    level = str(summary.get("level", ""))
    if level == "level1":
        summary["updated"] = False
        summary["reason"] = "level1_trial_training_no_formal_update"
        return summary

    if not summary.get("formal_update_allowed", False):
        summary["updated"] = False
        summary["reason"] = "model_update_not_allowed_by_run_mode"
        return summary

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.lgbm_path)
    summary["updated"] = True
    summary["path"] = str(paths.lgbm_path)
    summary["reason"] = "formal_model_updated"
    return summary


def train_step5_models(s, lookback: int = 150, theme_file_name: str = "step4_theme.csv") -> Dict[str, Any]:
    res_lr = train_step5_lr(s, lookback=lookback, theme_file_name=theme_file_name)
    res_lgbm = train_step5_lgbm(s, lookback=lookback, theme_file_name=theme_file_name)
    return {
        "ok": bool(res_lr.get("ok")) or bool(res_lgbm.get("ok")),
        "updated": bool(res_lr.get("updated")) or bool(res_lgbm.get("updated")),
        "run_mode": _resolve_run_mode(),
        "lr": res_lr,
        "lgbm": res_lgbm,
    }


# =========================================================
# Probability layers
# =========================================================

def _calc_prob_rule(df: pd.DataFrame) -> pd.Series:
    """
    规则概率：
    - 只用于主模型缺失 / 融合补充
    - 内部用数值兜底，不反写上游契约字段
    """
    feat = _ensure_inference_input(df)

    strength = _clip01(feat["StrengthScore"] / 100.0)
    theme = _clip01(feat["ThemeBoost"])  # ThemeBoost 本身就是 0~1

    turnover = feat["turnover_rate"].astype(float)
    turnover_score = pd.Series(
        np.where(turnover <= 0, 0.0, np.exp(-((turnover - 18.0) / 12.0) ** 2)),
        index=df.index,
        dtype="float64",
    )
    turnover_score = _clip01(turnover_score)

    seal_amount = feat["seal_amount"].astype(float)
    seal_score = _clip01(np.log1p(np.maximum(seal_amount, 0.0)) / 16.0)

    open_times = feat["open_times"].astype(float)
    open_score = _clip01(1.0 - np.minimum(np.maximum(open_times, 0.0), 8.0) / 8.0)

    rule = (
        0.42 * strength
        + 0.18 * theme
        + 0.16 * seal_score
        + 0.14 * open_score
        + 0.10 * turnover_score
    )
    return _clip01(rule)


def _predict_with_model(model: Any, X: np.ndarray, index: pd.Index) -> pd.Series:
    if model is None:
        return pd.Series([np.nan] * len(index), index=index, dtype="float64")
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X)
        return _clip01(pd.Series(proba, index=index, dtype="float64"))
    except Exception:
        return pd.Series([np.nan] * len(index), index=index, dtype="float64")


def _calc_prob_ml(df: pd.DataFrame, s=None) -> Tuple[pd.Series, pd.Series]:
    feat = _ensure_inference_input(df)
    X = feat[FEATURES].astype(float).values

    lr_model = load_lr(s)
    lgbm_model = load_lgbm(s)

    prob_lr = _predict_with_model(lr_model, X, df.index)
    prob_lgbm = _predict_with_model(lgbm_model, X, df.index)
    return prob_lr, prob_lgbm


def _fuse_probabilities(
    prob_rule: pd.Series,
    prob_lr: pd.Series,
    prob_lgbm: pd.Series,
    s=None,
) -> Tuple[pd.Series, pd.Series]:
    cfg = _get_ml_cfg(s)
    model_pref = str(cfg.get("model", "auto")).lower()
    fusion_mode = str(cfg.get("fusion_mode", "ml_first")).lower()
    fallback_to_rule = bool(cfg.get("fallback_to_rule", True))
    w_rule = float(cfg.get("rule_weight", 0.30))
    w_ml = float(cfg.get("ml_weight", 0.70))

    w_sum = max(1e-12, w_rule + w_ml)
    w_rule /= w_sum
    w_ml /= w_sum

    # 选择主 ML 源
    if model_pref == "logistic":
        ml_main = prob_lr
        ml_name = "lr"
    elif model_pref == "lightgbm":
        ml_main = prob_lgbm
        ml_name = "lgbm"
    else:
        # auto: 优先 lgbm，再 lr
        lgbm_ok = prob_lgbm.notna()
        lr_ok = prob_lr.notna()
        ml_main = prob_lgbm.where(lgbm_ok, prob_lr)
        # 每行动态来源
        ml_name = None

    Probability = pd.Series([np.nan] * len(prob_rule), index=prob_rule.index, dtype="float64")
    src = pd.Series(["unresolved"] * len(prob_rule), index=prob_rule.index, dtype="object")

    if model_pref in {"logistic", "lightgbm"}:
        ml_ok = ml_main.notna()

        if fusion_mode == "weighted":
            fused = w_rule * prob_rule.fillna(0.0) + w_ml * ml_main.fillna(prob_rule.fillna(0.0))
            Probability[:] = fused
            src[:] = np.where(ml_ok, f"blend:{ml_name}+rule", "fallback_rule")
        else:
            if fallback_to_rule:
                Probability[:] = np.where(ml_ok, ml_main, prob_rule)
                src[:] = np.where(ml_ok, ml_name, "fallback_rule")
            else:
                Probability[:] = ml_main
                src[:] = np.where(ml_ok, ml_name, "ml_missing")

    else:
        # auto 模式：优先 lgbm，次选 lr，最后 rule
        lgbm_ok = prob_lgbm.notna()
        lr_ok = prob_lr.notna()

        if fusion_mode == "weighted":
            # auto-weighted：优先选可用主 ML，再和 rule 融合
            ml_any = prob_lgbm.where(lgbm_ok, prob_lr)
            Probability[:] = w_rule * prob_rule.fillna(0.0) + w_ml * ml_any.fillna(prob_rule.fillna(0.0))
            src[:] = np.where(
                lgbm_ok,
                "blend:lgbm+rule",
                np.where(lr_ok, "blend:lr+rule", "fallback_rule"),
            )
        else:
            if fallback_to_rule:
                Probability[:] = np.where(
                    lgbm_ok,
                    prob_lgbm,
                    np.where(lr_ok, prob_lr, prob_rule),
                )
                src[:] = np.where(
                    lgbm_ok,
                    "lgbm",
                    np.where(lr_ok, "lr", "fallback_rule"),
                )
            else:
                Probability[:] = np.where(lgbm_ok, prob_lgbm, prob_lr)
                src[:] = np.where(lgbm_ok, "lgbm", np.where(lr_ok, "lr", "ml_missing"))

    Probability = _clip01(Probability.fillna(prob_rule.fillna(0.0)))
    return Probability.astype("float64"), src.astype("object")


# =========================================================
# feature_history append (compat)
# =========================================================

def _write_feature_history(
    raw_input_df: pd.DataFrame,
    out_df: pd.DataFrame,
    trade_date: str,
    s=None,
) -> Dict[str, Any]:
    """
    当前仍保留 Step5 对 feature_history 的兼容写入。
    后续 writers.py 收口后，这里仍可继续保留，但不应与 writers 冲突。
    """
    try:
        outputs_dir = _get_outputs_dir(s)
        raw_input_df = _normalize_id_columns(_ensure_df(raw_input_df))
        out_df = _ensure_df(out_df)

        if raw_input_df.empty or "ts_code" not in raw_input_df.columns:
            return {"ok": False, "reason": "raw_input invalid"}

        raw_input_df = _backfill_features_from_step3(raw_input_df, trade_date=trade_date, outputs_dir=outputs_dir)

        tmp = pd.DataFrame(index=raw_input_df.index)
        tmp["trade_date"] = str(trade_date).strip()
        tmp["ts_code"] = raw_input_df["ts_code"].astype(str).str.strip()

        name_col = _get_name_col(raw_input_df)
        tmp["name"] = raw_input_df[name_col].astype(str) if name_col else ""

        for c in FEATURES:
            if c in raw_input_df.columns:
                tmp[c] = _to_numeric_nullable(raw_input_df[c])
            else:
                tmp[c] = np.nan

        for c in ["prob_lr", "prob_lgbm", "prob_rule", "Probability", "_prob_src"]:
            if c in out_df.columns:
                tmp[c] = out_df[c]

        tmp["run_time_utc"] = _utc_now_iso()
        tmp["run_mode"] = _resolve_run_mode()

        base = outputs_dir / "learning"
        _ensure_dir(base)
        fp = base / "feature_history.csv"

        if fp.exists():
            try:
                old = pd.read_csv(fp)
            except Exception:
                old = pd.DataFrame()
            merged = pd.concat([old, tmp], ignore_index=True, sort=False)
        else:
            merged = tmp

        merged["trade_date"] = merged.get("trade_date", "").astype(str).str.strip()
        merged["ts_code"] = merged.get("ts_code", "").astype(str).str.strip()
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

        if "Probability" in merged.columns:
            merged["_sort_prob"] = pd.to_numeric(merged["Probability"], errors="coerce").fillna(0.0)
            merged = merged.sort_values(["trade_date", "_sort_prob"], ascending=[True, False])
            merged = merged.drop(columns=["_sort_prob"], errors="ignore")

        merged.to_csv(fp, index=False, encoding="utf-8")
        return {"ok": True, "path": str(fp), "rows": int(len(merged))}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# =========================================================
# Main inference
# =========================================================

def run_step5(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    raw_input = _ensure_df(theme_df)
    if raw_input.empty:
        out = pd.DataFrame()
        for c in ["prob_lr", "prob_lgbm", "prob_rule", "Probability", "_prob_src"]:
            out[c] = pd.Series(dtype="float64" if c != "_prob_src" else "object")
        return out

    trade_date = _guess_trade_date(raw_input)
    outputs_dir = _get_outputs_dir(s)

    out = _normalize_id_columns(raw_input)
    out = _backfill_features_from_step3(out, trade_date=trade_date, outputs_dir=outputs_dir)

    cfg = _get_ml_cfg(s)
    enable_rule = bool(cfg.get("enable_rule", True))
    enable_ml = bool(cfg.get("enable_ml", True))

    if enable_rule:
        out["prob_rule"] = _calc_prob_rule(out)
    else:
        out["prob_rule"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")

    if enable_ml:
        prob_lr, prob_lgbm = _calc_prob_ml(out, s=s)
        out["prob_lr"] = prob_lr
        out["prob_lgbm"] = prob_lgbm
    else:
        out["prob_lr"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")
        out["prob_lgbm"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")

    Probability, _prob_src = _fuse_probabilities(
        prob_rule=pd.to_numeric(out["prob_rule"], errors="coerce"),
        prob_lr=pd.to_numeric(out["prob_lr"], errors="coerce"),
        prob_lgbm=pd.to_numeric(out["prob_lgbm"], errors="coerce"),
        s=s,
    )

    out["Probability"] = Probability.astype("float64")
    out["_prob_src"] = _prob_src.astype("object")
    out["run_mode"] = _resolve_run_mode()
    out["run_time_utc"] = _utc_now_iso()

    clip_min = float(cfg.get("clip_min", 0.0))
    clip_max = float(cfg.get("clip_max", 1.0))
    for c in ["prob_lr", "prob_lgbm", "prob_rule", "Probability"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(clip_min, clip_max)

    _write_feature_history(raw_input_df=raw_input, out_df=out, trade_date=trade_date, s=s)

    sort_cols = ["Probability"]
    if "StrengthScore" in out.columns:
        sort_cols.append("StrengthScore")
    ascending = [False] * len(sort_cols)
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

    prefer_order = [
        "trade_date",
        "ts_code",
        "name",
        "StrengthScore",
        "ThemeBoost",
        "seal_amount",
        "open_times",
        "turnover_rate",
        "prob_lr",
        "prob_lgbm",
        "prob_rule",
        "Probability",
        "_prob_src",
        "run_mode",
        "run_time_utc",
    ]
    exist = [c for c in prefer_order if c in out.columns]
    others = [c for c in out.columns if c not in exist]
    out = out[exist + others]

    return out


def run(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(theme_df=theme_df, s=s)
