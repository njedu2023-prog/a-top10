#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : 概率模型推断（ML核心层）—— Logistic + LightGBM 双模型 + 可训练闭环

输入：
    theme_df（step4输出，至少应包含 FEATURES 所需字段）

输出：
    prob_df（附带 Probability / Probability_lr / Probability_lgb / _prob_src 等）

闭环：
    - 训练数据：历史若干天的 step4 输出（建议落盘为 step4_theme.csv）
    - 标签：next_day 是否涨停（用 next_day snapshot 的 limit_list_d.csv 来打标）
    - 模型：LogisticRegression + LightGBM（真实训练/推断）
    - 持久化：
        models/step5_lr.joblib
        models/step5_lgb.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# joblib (for persistence)
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# lightgbm (optional but recommended)
try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None


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

DEFAULT_THEMEBOOST = 1.0  # ThemeBoost 合理默认值应接近 1


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
    """
    保证 FEATURES 存在 + 变成 float + 清洗 nan/inf
    同时对 ThemeBoost 给更合理的默认值 1.0
    """
    out = df.copy()

    for c in FEATURES:
        if c not in out.columns:
            out[c] = DEFAULT_THEMEBOOST if c == "ThemeBoost" else 0.0

        default = DEFAULT_THEMEBOOST if c == "ThemeBoost" else 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    return out


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ID_COL_CANDIDATES)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_float_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    x = df[cols].astype(float).values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# -------------------------
# Model IO
# -------------------------
@dataclass
class Step5ModelPaths:
    base_dir: Path
    lr_path: Path
    lgb_path: Path


def _get_model_paths(s=None) -> Step5ModelPaths:
    """
    约定：
    - 若 Settings 里有 s.data_repo.models_dir / model_dir / root 更好；
    - 否则落到当前目录 ./models
    """
    base: Optional[Path] = None

    if s is not None:
        data_repo = getattr(s, "data_repo", None)

        # 1) data_repo.models_dir/model_dir
        if data_repo is not None:
            for attr in ["models_dir", "model_dir"]:
                if hasattr(data_repo, attr):
                    try:
                        base = Path(getattr(data_repo, attr))
                        break
                    except Exception:
                        pass

            # 2) data_repo.root/models
            if base is None and hasattr(data_repo, "root"):
                try:
                    base = Path(getattr(data_repo, "root")) / "models"
                except Exception:
                    base = None

    if base is None:
        base = Path("models")

    base.mkdir(parents=True, exist_ok=True)
    return Step5ModelPaths(
        base_dir=base,
        lr_path=base / "step5_lr.joblib",
        lgb_path=base / "step5_lgb.joblib",
    )


def load_lr(s=None) -> Optional[Pipeline]:
    paths = _get_model_paths(s)
    if joblib is None or (not paths.lr_path.exists()):
        return None
    try:
        return joblib.load(paths.lr_path)
    except Exception:
        return None


def save_lr(model: Pipeline, s=None) -> None:
    if joblib is None:
        return
    paths = _get_model_paths(s)
    try:
        joblib.dump(model, paths.lr_path)
    except Exception:
        return


def load_lgb(s=None):
    paths = _get_model_paths(s)
    if joblib is None or (not paths.lgb_path.exists()):
        return None
    try:
        return joblib.load(paths.lgb_path)
    except Exception:
        return None


def save_lgb(model, s=None) -> None:
    if joblib is None:
        return
    paths = _get_model_paths(s)
    try:
        joblib.dump(model, paths.lgb_path)
    except Exception:
        return


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
        X = _safe_float_matrix(feat_df, FEATURES)

        # 过滤：全0行去掉（无信息）
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


def _list_snapshot_dirs_from_settings(s, lookback_plus1: int) -> List[Tuple[str, Path]]:
    """
    兼容你们的 data_repo：
    - 优先：list_snapshot_dates() + snapshot_dir(date)
    - 兜底：扫描 data_repo.root/snapshots/*
    """
    if s is None or not hasattr(s, "data_repo"):
        return []

    dr = s.data_repo
    dates: List[str] = []

    if hasattr(dr, "list_snapshot_dates"):
        try:
            dates = list(dr.list_snapshot_dates())
        except Exception:
            dates = []
    else:
        root = getattr(dr, "root", None)
        if root is None:
            return []
        snap_root = Path(root) / "snapshots"
        if not snap_root.exists():
            return []
        dates = sorted([p.name for p in snap_root.iterdir() if p.is_dir()])

    dates = [d for d in dates if isinstance(d, str) and len(d) >= 8]
    dates = dates[-lookback_plus1:]

    snapshot_dirs: List[Tuple[str, Path]] = []
    for d in dates:
        snap = None
        if hasattr(dr, "snapshot_dir"):
            try:
                snap = dr.snapshot_dir(d)
            except Exception:
                snap = None
        if snap is None:
            # fallback：root/snapshots/d
            root = getattr(dr, "root", None)
            if root is None:
                continue
            snap = Path(root) / "snapshots" / d

        p = Path(snap)
        if p.exists():
            snapshot_dirs.append((d, p))

    snapshot_dirs.sort(key=lambda x: x[0])
    return snapshot_dirs


# -------------------------
# Train: Logistic + LightGBM
# -------------------------
def train_step5_models(
    s,
    lookback: int = 120,
    theme_file_name: str = "step4_theme.csv",
    min_pos: int = 30,
    min_samples: int = 300,
    lgb_params: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    每天收盘后调用：
    - 从数据仓库取最近 lookback 天 snapshot
    - 用 step4_theme.csv 做特征，次日 limit_list_d.csv 打标签
    - 训练 LR + LGB（若 LightGBM 不可用，会跳过但不报错）
    - 持久化
    """
    if s is None or not hasattr(s, "data_repo"):
        return {"ok": False, "reason": "missing Settings/data_repo"}

    snapshot_dirs = _list_snapshot_dirs_from_settings(s, lookback_plus1=lookback + 1)
    if len(snapshot_dirs) < 10:
        return {"ok": False, "reason": f"not enough snapshots: {len(snapshot_dirs)}"}

    X, y = _build_xy_from_history(snapshot_dirs, theme_file_name=theme_file_name)

    pos = int(y.sum()) if len(y) else 0
    if len(y) < min_samples or pos < min_pos:
        return {"ok": False, "reason": f"not enough labeled samples: n={len(y)}, pos={pos}"}

    result: Dict[str, Any] = {"ok": True, "n": int(len(y)), "pos": pos, "lookback": lookback}

    # ---- 1) Logistic Regression ----
    lr_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    lr_model.fit(X, y)
    save_lr(lr_model, s=s)
    result["lr"] = "trained"

    # ---- 2) LightGBM ----
    if lgb is None:
        result["lgb"] = "skipped(lightgbm_not_installed)"
        return result

    # 默认参数（偏稳健，不追求极致）
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    if isinstance(lgb_params, dict):
        params.update(lgb_params)

    try:
        lgb_model = lgb.LGBMClassifier(**params)
        lgb_model.fit(X, y)
        save_lgb(lgb_model, s=s)
        result["lgb"] = "trained"
    except Exception as e:
        # 不让训练失败把整个链路炸掉
        result["lgb"] = f"failed({type(e).__name__})"

    return result


# -------------------------
# Inference: Logistic + LightGBM (ensemble)
# -------------------------
def run_step5(
    theme_df: pd.DataFrame,
    s=None,
    ensemble_w_lr: float = 0.5,
    ensemble_w_lgb: float = 0.5,
) -> pd.DataFrame:
    """
    推断（严格保证能跑通）：
    1) 若已保存 LR/LGB，则用真实 predict_proba
    2) 若某个模型缺失，则自动只用另一个
    3) 若两个都缺失，则用 pseudo-sigmoid 兜底（保证链路不断）
    """
    out = _ensure_features(theme_df)

    X = _safe_float_matrix(out, FEATURES)

    lr_model = load_lr(s=s)
    lgb_model = load_lgb(s=s)

    p_lr = None
    p_lgb = None

    if lr_model is not None:
        try:
            p_lr = np.clip(lr_model.predict_proba(X)[:, 1], 0.0, 1.0)
            out["Probability_lr"] = p_lr
        except Exception:
            p_lr = None

    if lgb_model is not None:
        try:
            # lightgbm/sklearn API
            p_lgb = np.clip(lgb_model.predict_proba(X)[:, 1], 0.0, 1.0)
            out["Probability_lgb"] = p_lgb
        except Exception:
            p_lgb = None

    # ---- ensemble logic ----
    if (p_lr is not None) and (p_lgb is not None):
        w1 = float(ensemble_w_lr)
        w2 = float(ensemble_w_lgb)
        if (w1 + w2) <= 0:
            w1, w2 = 0.5, 0.5
        prob = (w1 * p_lr + w2 * p_lgb) / (w1 + w2)
        out["Probability"] = np.clip(prob, 0.0, 1.0)
        out["_prob_src"] = "lr+lgb"

    elif p_lr is not None:
        out["Probability"] = np.clip(p_lr, 0.0, 1.0)
        out["_prob_src"] = "lr_only"

    elif p_lgb is not None:
        out["Probability"] = np.clip(p_lgb, 0.0, 1.0)
        out["_prob_src"] = "lgb_only"

    else:
        # 兜底 pseudo probability (sigmoid)
        z = (
            0.04 * out["StrengthScore"].astype(float).values
            + 1.4 * out["ThemeBoost"].astype(float).values
            - 0.35 * out["open_times"].astype(float).values
            + 0.000000005 * out["seal_amount"].astype(float).values
        )
        out["Probability"] = _sigmoid(z)
        out["_prob_src"] = "pseudo"

    return out.sort_values("Probability", ascending=False)


# Backward-compatible alias：主程序若统一用 run() 调用每一步，这里也给出
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(df, s=s)


if __name__ == "__main__":
    print("Step5 Probability model loaded.")
