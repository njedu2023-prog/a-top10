#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : 概率模型推断（ML核心层） + 可训练LR闭环

输入：
    theme_df（step4输出，至少应包含 FEATURES 所需字段）

输出：
    prob_df（附带 Probability / _prob 等）

闭环：
    - 训练数据：历史若干天的 step4 输出（建议落盘为 step4_theme.csv）
    - 标签：next_day 是否涨停（用 next_day 的 limit_list_d.csv 来打标）
    - 模型：LogisticRegression（可扩展）
    - 持久化：models/step5_lr.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -------------------------
# Feature Columns (真实字段)
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


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float) -> pd.Series:
    if (col is None) or (col not in df.columns):
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ID_COL_CANDIDATES)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------
# Model IO
# -------------------------
@dataclass
class Step5ModelPaths:
    model_path: Path
    meta_path: Path


def _get_model_paths(s=None) -> Step5ModelPaths:
    """
    约定：
    - 若 Settings 里有 s.data_repo.root / s.data_repo.models_dir 更好；
    - 这里做兼容：优先 s.data_repo.xxx，否则落到当前目录 ./models
    """
    base = None
    if s is not None:
        # 你们的 Settings 可能不同，这里尽量宽松取路径
        for attr in ["models_dir", "model_dir"]:
            if hasattr(getattr(s, "data_repo", object()), attr):
                base = Path(getattr(s.data_repo, attr))
                break
        if base is None and hasattr(getattr(s, "data_repo", object()), "root"):
            base = Path(s.data_repo.root) / "models"

    if base is None:
        base = Path("models")

    base.mkdir(parents=True, exist_ok=True)
    return Step5ModelPaths(
        model_path=base / "step5_lr.joblib",
        meta_path=base / "step5_lr_meta.json",
    )


def load_model(s=None) -> Optional[Pipeline]:
    paths = _get_model_paths(s)
    if joblib is None:
        return None
    if not paths.model_path.exists():
        return None
    try:
        return joblib.load(paths.model_path)
    except Exception:
        return None


def save_model(model: Pipeline, s=None) -> None:
    paths = _get_model_paths(s)
    if joblib is None:
        return
    joblib.dump(model, paths.model_path)


# -------------------------
# Training Data Builder (闭环核心)
# -------------------------
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


def _label_from_next_day_limit_list(next_snap: Path) -> set:
    """
    用 next_day 的 limit_list_d.csv 打标签：在涨停列表里 => y=1
    你也可以替换成你们更准确的“次日涨停”口径文件。
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
    s=None,
    theme_file_name: str = "step4_theme.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    snapshot_dirs: [(trade_date, snap_path)]，按日期升序
    对每一天 d，用 d 的 step4_theme.csv 作为特征，用 d+1 的涨停列表打标签
    """
    X_rows = []
    y_rows = []

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

        # 基本过滤：全0行去掉（无信息）
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


def train_step5_lr(
    s,
    lookback: int = 90,
    theme_file_name: str = "step4_theme.csv",
    min_pos: int = 30,
    min_samples: int = 300,
) -> dict:
    """
    每天收盘后调用：
    - 从数据仓库取最近 lookback 天 snapshot（你们仓库结构不同的话，改这里的取日期方式）
    - 训练 LR 并持久化
    """
    if s is None or not hasattr(s, "data_repo"):
        return {"ok": False, "reason": "missing Settings/data_repo"}

    # 约定 data_repo 有 list_snapshot_dates() / snapshot_dir(date)
    # 如果你们没有 list_snapshot_dates()，就自己扫目录改一下
    dates: List[str] = []
    if hasattr(s.data_repo, "list_snapshot_dates"):
        dates = list(s.data_repo.list_snapshot_dates())
    else:
        # 兜底：尝试扫描 snapshot 根目录（你们可按实际改）
        root = getattr(s.data_repo, "root", None)
        if root is None:
            return {"ok": False, "reason": "data_repo has no list_snapshot_dates/root"}
        snap_root = Path(root) / "snapshots"
        if not snap_root.exists():
            return {"ok": False, "reason": f"snapshot root not found: {snap_root}"}
        dates = sorted([p.name for p in snap_root.iterdir() if p.is_dir()])

    dates = [d for d in dates if isinstance(d, str) and len(d) >= 8]
    dates = dates[-(lookback + 1) :]  # 需要 next_day 打标签，所以 +1

    snapshot_dirs: List[Tuple[str, Path]] = []
    for d in dates:
        snap = s.data_repo.snapshot_dir(d) if hasattr(s.data_repo, "snapshot_dir") else None
        if snap is None:
            continue
        snapshot_dirs.append((d, Path(snap)))

    snapshot_dirs = [(d, p) for (d, p) in snapshot_dirs if p.exists()]
    if len(snapshot_dirs) < 10:
        return {"ok": False, "reason": "not enough snapshots"}

    X, y = _build_xy_from_history(snapshot_dirs, s=s, theme_file_name=theme_file_name)

    pos = int(y.sum()) if len(y) else 0
    if len(y) < min_samples or pos < min_pos:
        return {"ok": False, "reason": f"not enough labeled samples: n={len(y)}, pos={pos}"}

    # 标准化 + LR（class_weight 解决正样本稀少）
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    model.fit(X, y)

    save_model(model, s=s)
    return {"ok": True, "n": int(len(y)), "pos": pos, "lookback": lookback}


# -------------------------
# Inference
# -------------------------
def run_step5(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    推断：
    1) 若本地已有训练模型，优先用 LR predict_proba
    2) 否则用 pseudo-sigmoid 兜底
    """
    out = _ensure_features(theme_df)

    model = load_model(s=s)
    if model is not None:
        X = out[FEATURES].astype(float).values
        proba = model.predict_proba(X)[:, 1]
        out["Probability"] = np.clip(proba, 0.0, 1.0)
        out["_prob_src"] = "lr"
    else:
        # 兜底 pseudo probability (sigmoid)
        z = (
            0.04 * out["StrengthScore"].astype(float).values
            + 1.5 * out["ThemeBoost"].astype(float).values
            - 0.3 * out["open_times"].astype(float).values
        )
        out["Probability"] = _sigmoid(z)
        out["_prob_src"] = "pseudo"

    return out.sort_values("Probability", ascending=False)


# Backward-compatible alias：主程序若统一用 run() 调用每一步，这里也给出
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(df, s=s)


if __name__ == "__main__":
    print("Step5 Probability model loaded.")
