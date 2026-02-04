#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : 概率模型推断（ML核心层） + 可训练闭环（LR + LightGBM）

输入：
    theme_df（step4输出，至少应包含 FEATURES 所需字段）
输出：
    prob_df（附带 Probability / _prob_src 等）

闭环训练：
    - 训练数据：历史若干天的 step4 输出（建议落盘为 step4_theme.csv）
    - 标签：next_day 是否涨停（用 next_day 的 limit_list_d.csv 来打标）
    - 模型：
        1) LogisticRegression（标准化 + class_weight=balanced）
        2) LightGBM（LGBMClassifier，处理非线性更强）
    - 持久化：
        models/step5_lr.joblib
        models/step5_lgbm.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Dict, Any

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
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


# -------------------------
# Feature Columns（真实字段）
# -------------------------
FEATURES = [
    "StrengthScore",
    "ThemeBoost",
    "seal_amount",
    "open_times",
    "turnover_rate",
]

ID_COL_CANDIDATES = ["ts_code", "code", "TS_CODE"]


# -------------------------
# Helpers
# -------------------------
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


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ID_COL_CANDIDATES)


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float64")
        )
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _read_csv_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(p, dtype=str, encoding="gbk")
        except Exception:
            return pd.DataFrame()


# -------------------------
# Model IO
# -------------------------
@dataclass
class Step5ModelPaths:
    base: Path
    lr_path: Path
    lgbm_path: Path


def _get_model_paths(s=None) -> Step5ModelPaths:
    """
    约定：
    - 若 Settings 里有 s.data_repo.models_dir / model_dir / root 更好；
    - 这里做兼容：优先 s.data_repo.xxx，否则落到当前目录 ./models
    """
    base = None
    if s is not None and hasattr(s, "data_repo"):
        dr = s.data_repo
        for attr in ["models_dir", "model_dir"]:
            if hasattr(dr, attr):
                base = Path(getattr(dr, attr))
                break
        if base is None and hasattr(dr, "root"):
            base = Path(dr.root) / "models"

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
        return


def load_lr(s=None) -> Optional[Pipeline]:
    paths = _get_model_paths(s)
    return _load_joblib(paths.lr_path)


def save_lr(model: Pipeline, s=None) -> None:
    paths = _get_model_paths(s)
    _save_joblib(model, paths.lr_path)


def load_lgbm(s=None):
    paths = _get_model_paths(s)
    return _load_joblib(paths.lgbm_path)


def save_lgbm(model, s=None) -> None:
    paths = _get_model_paths(s)
    _save_joblib(model, paths.lgbm_path)


# -------------------------
# Labeling / Training Data Builder
# -------------------------
def _label_from_next_day_limit_list(next_snap: Path) -> set:
    """
    用 next_day 的 limit_list_d.csv 打标签：在涨停列表里 => y=1
    """
    ll = _read_csv_if_exists(next_snap / "limit_list_d.csv")
    if ll.empty:
        return set()
    ts_col = _get_ts_code_col(ll)
    if ts_col is None:
        return set()
    return set(ll[ts_col].astype(str).tolist())


def _build_xy_from_history(
    snapshot_dirs: List[Tuple[str, Path]],
    theme_file_name: str = "step4_theme.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    snapshot_dirs: [(trade_date, snap_path)]，按日期升序
    对每一天 d，用 d 的 step4_theme.csv 作为特征，用 d+1 的涨停列表打标签
    """
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []

    for i in range(len(snapshot_dirs) - 1):
        d, snap = snapshot_dirs[i]
        d_next, snap_next = snapshot_dirs[i + 1]

        feat_df = _read_csv_if_exists(snap / theme_file_name)
        if feat_df.empty:
            continue

        ts_col = _get_ts_code_col(feat_df)
        if ts_col is None:
            continue

        feat_df = _ensure_features(feat_df)
        limit_set = _label_from_next_day_limit_list(snap_next)

        y = feat_df[ts_col].astype(str).apply(lambda x: 1 if x in limit_set else 0).astype(int).values
        X = feat_df[FEATURES].astype(float).values

        mask = np.isfinite(X).all(axis=1) & (np.abs(X).sum(axis=1) > 0)
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            continue

        X_rows.append(X)
        y_rows.append(y)

    if not X_rows:
        return np.zeros((0, len(FEATURES))), np.zeros((0,), dtype=int)

    X_all = np.vstack(X_rows)
    y_all = np.concatenate(y_rows)
    return X_all, y_all


def _resolve_snapshot_dirs_from_settings(s, lookback: int) -> List[Tuple[str, Path]]:
    """
    从 s.data_repo 获取最近 lookback+1 天快照目录（+1 用于 next_day 打标）
    要求：s.data_repo 至少提供 snapshot_dir(date)；可选 list_snapshot_dates()
    """
    if s is None or not hasattr(s, "data_repo"):
        return []

    dr = s.data_repo
    dates: List[str] = []

    if hasattr(dr, "list_snapshot_dates"):
        dates = list(dr.list_snapshot_dates())
    else:
        # 兜底：扫描你仓库本地快照根目录（按你当前仓库结构：_warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD）
        # 如果你们 data_repo.py 已经封装了 snapshot_dir，就尽量走 snapshot_dir(date)
        root = getattr(dr, "warehouse_root", None)
        repo_name = getattr(dr, "repo_name", None)
        raw_dir = getattr(dr, "raw_dir", None)

        if root and repo_name and raw_dir:
            raw_root = Path(root) / repo_name / raw_dir
            if raw_root.exists():
                # 扫两级：YYYY / YYYYMMDD
                tmp = []
                for ydir in raw_root.iterdir():
                    if not ydir.is_dir():
                        continue
                    for ddir in ydir.iterdir():
                        if ddir.is_dir() and len(ddir.name) >= 8:
                            tmp.append(ddir.name)
                dates = sorted(tmp)
        else:
            # 最后一层兜底：尝试 data_repo/snapshots（你之前也建了这个目录）
            base = Path("data_repo/snapshots")
            if base.exists():
                dates = sorted([p.name for p in base.iterdir() if p.is_dir()])

    dates = [d for d in dates if isinstance(d, str) and len(d) >= 8]
    dates = dates[-(lookback + 1):]

    out: List[Tuple[str, Path]] = []
    for d in dates:
        if hasattr(dr, "snapshot_dir"):
            p = Path(dr.snapshot_dir(d))
        else:
            # 若没有 snapshot_dir，只能按 data_repo/snapshots/date 兜底
            p = Path("data_repo/snapshots") / d
        if p.exists():
            out.append((d, p))

    return out


# -------------------------
# Training (LR + LGBM)
# -------------------------
def train_step5_lr(
    s,
    lookback: int = 90,
    theme_file_name: str = "step4_theme.csv",
    min_pos: int = 30,
    min_samples: int = 300,
) -> Dict[str, Any]:
    snapshot_dirs = _resolve_snapshot_dirs_from_settings(s, lookback=lookback)
    if len(snapshot_dirs) < 10:
        return {"ok": False, "model": "lr", "reason": "not enough snapshots"}

    X, y = _build_xy_from_history(snapshot_dirs, theme_file_name=theme_file_name)

    pos = int(y.sum()) if len(y) else 0
    if len(y) < min_samples or pos < min_pos:
        return {"ok": False, "model": "lr", "reason": f"not enough labeled samples: n={len(y)}, pos={pos}"}

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    model.fit(X, y)
    save_lr(model, s=s)
    return {"ok": True, "model": "lr", "n": int(len(y)), "pos": pos, "lookback": lookback}


def train_step5_lgbm(
    s,
    lookback: int = 120,
    theme_file_name: str = "step4_theme.csv",
    min_pos: int = 30,
    min_samples: int = 300,
) -> Dict[str, Any]:
    if lgb is None:
        return {"ok": False, "model": "lgbm", "reason": "lightgbm not installed"}

    snapshot_dirs = _resolve_snapshot_dirs_from_settings(s, lookback=lookback)
    if len(snapshot_dirs) < 10:
        return {"ok": False, "model": "lgbm", "reason": "not enough snapshots"}

    X, y = _build_xy_from_history(snapshot_dirs, theme_file_name=theme_file_name)

    pos = int(y.sum()) if len(y) else 0
    if len(y) < min_samples or pos < min_pos:
        return {"ok": False, "model": "lgbm", "reason": f"not enough labeled samples: n={len(y)}, pos={pos}"}

    # 轻量默认参数（先跑通闭环，后面再细调）
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    save_lgbm(model, s=s)
    return {"ok": True, "model": "lgbm", "n": int(len(y)), "pos": pos, "lookback": lookback}


def train_step5_models(
    s,
    lookback: int = 120,
    theme_file_name: str = "step4_theme.csv",
) -> Dict[str, Any]:
    """
    统一入口：每天收盘后先训练（尽量两种都训）
    """
    res_lr = train_step5_lr(s, lookback=max(60, int(lookback * 0.75)), theme_file_name=theme_file_name)
    res_lgbm = train_step5_lgbm(s, lookback=lookback, theme_file_name=theme_file_name)
    ok_any = bool(res_lr.get("ok")) or bool(res_lgbm.get("ok"))
    return {"ok": ok_any, "lr": res_lr, "lgbm": res_lgbm}


# -------------------------
# Inference
# -------------------------
def run_step5(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    推断优先级：
    1) LightGBM（若存在已训练模型）
    2) LogisticRegression（若存在已训练模型）
    3) pseudo-sigmoid（兜底）
    """
    out = _ensure_features(theme_df)
    X = out[FEATURES].astype(float).values

    m_lgbm = load_lgbm(s=s)
    if m_lgbm is not None:
        try:
            proba = m_lgbm.predict_proba(X)[:, 1]
            out["Probability"] = np.clip(proba, 0.0, 1.0)
            out["_prob_src"] = "lgbm"
            return out.sort_values("Probability", ascending=False)
        except Exception:
            # 若模型损坏/不兼容，继续降级
            pass

    m_lr = load_lr(s=s)
    if m_lr is not None:
        try:
            proba = m_lr.predict_proba(X)[:, 1]
            out["Probability"] = np.clip(proba, 0.0, 1.0)
            out["_prob_src"] = "lr"
            return out.sort_values("Probability", ascending=False)
        except Exception:
            pass

    # 兜底 pseudo probability
    z = (
        0.04 * out["StrengthScore"].astype(float).values
        + 1.5 * out["ThemeBoost"].astype(float).values
        - 0.3 * out["open_times"].astype(float).values
    )
    out["Probability"] = _sigmoid(z)
    out["_prob_src"] = "pseudo"
    return out.sort_values("Probability", ascending=False)


def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(df, s=s)


if __name__ == "__main__":
    print("Step5 Probability model loaded.")
