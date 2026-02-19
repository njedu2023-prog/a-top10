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

✅ 本终版新增（自学习最后一公里关键）：
- 每个交易日推断后，自动落盘/追加：outputs/learning/feature_history.csv
  用于 Step7 进行 next_day 打标与训练样本构建。
"""

from __future__ import annotations

import os
import hashlib
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

# 字段别名映射：不同步骤文件输出的字段名不必一样，通过这里相互对应
FEATURE_ALIASES: Dict[str, Sequence[str]] = {
    "StrengthScore": [
        "StrengthScore", "strengthscore", "strength_score", "Strength", "strength",
        "强度得分", "强度", "强度分",
    ],
    "ThemeBoost": [
        "ThemeBoost", "themeboost", "theme_boost", "theme_boost_score",
        "theme_boost_value", "theme_boost_ratio",
        "题材加成", "题材加成分", "题材加成得分", "题材",
    ],
    "seal_amount": [
        "seal_amount", "sealamount", "seal_amt", "sealAmount", "seal",
        "封单金额", "封单",
    ],
    "open_times": [
        "open_times", "opentimes", "open_time", "openTimes", "open_count",
        "openings", "打开次数", "打开次", "开板次数",
    ],
    "turnover_rate": [
        "turnover_rate", "turnoverrate", "turnover", "turnoverRate",
        "换手率", "换手", "换手率%",
    ],
}

# id 别名（用于涨停打标 / 组装数据）
TS_CODE_ALIASES = [
    "ts_code", "TS_CODE", "tscode", "TScode",
    "stock_code", "symbol", "sec_code", "stk_code", "code",
    "证券代码", "股票代码", "代码",
]


# -------------------------
# Helpers
# -------------------------
def _ensure_df(obj) -> pd.DataFrame:
    """兼容：上游误传 dict/list（比如某一步返回了 dict），这里尽量兜底转 DataFrame。"""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()
    if isinstance(obj, (list, tuple)):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


def _lower_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将各种可能的字段名同一步进行基本的横向对齐。"""
    df = _ensure_df(df)
    if df.empty:
        return df

    out = df.copy()
    lower = _lower_map(out)

    for canonical, aliases in FEATURE_ALIASES.items():
        key = str(canonical).strip().lower()

        # 1) 若已经存在 canonical（或大小写同名），直接对齐到 canonical
        if key in lower:
            col = lower[key]
            if col != canonical:
                out[canonical] = out[col]
            continue

        # 2) 否则去 alias 里找
        found = False
        for alias in aliases:
            alias_key = str(alias).strip().lower()
            if alias_key in lower:
                out[canonical] = out[lower[alias_key]]
                found = True
                break

        # 3) 没找到就给 NaN，后面统一 fillna(0)
        if not found:
            out[canonical] = np.nan

    return out


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 id 列动态映射到 ts_code（尽量）。"""
    df = _ensure_df(df)
    if df.empty:
        return df

    out = df.copy()
    lower = _lower_map(out)

    ts_code = None
    for name in TS_CODE_ALIASES:
        key = str(name).strip().lower()
        if key in lower:
            ts_code = lower[key]
            break

    # 若找到了某个列（可能叫 code/证券代码），统一映射成 ts_code
    if ts_code is not None:
        out["ts_code"] = out[ts_code]

    return out


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    df = _ensure_df(df)
    if df.empty:
        return None
    lower = _lower_map(df)
    for name in TS_CODE_ALIASES:
        key = str(name).strip().lower()
        if key in lower:
            return lower[key]
    return None


def _to_nosuffix(ts: str) -> str:
    ts = str(ts).strip()
    if not ts:
        return ts
    return ts.split(".")[0]


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    只保证 FEATURES 存在且是 float。
    注意：这里**不负责保留其他列**，推断时请用 raw_input.copy() 保持全字段。
    """
    df = _ensure_df(df)
    if df.empty:
        out = pd.DataFrame()
        for c in FEATURES:
            out[c] = pd.Series(dtype="float64")
        return out

    out = _normalize_feature_columns(df)

    for c in FEATURES:
        if c not in out.columns:
            out[c] = np.nan

        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float64")
        )
    return out[FEATURES].copy()


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


def _safe_log1p_pos(x: np.ndarray) -> np.ndarray:
    """用于金额/成交额等“正值长尾”的压缩：log1p(max(x,0))"""
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0)
    return np.log1p(x)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _guess_trade_date(df: pd.DataFrame) -> str:
    """
    推断交易日（YYYYMMDD）优先级：
    1) df 中常见日期列（trade_date/date/日期 等）
    2) 环境变量 TRADE_DATE
    3) 今天日期（workflow TZ 控制；这里直接用本机日期兜底）
    """
    df = _ensure_df(df)
    candidates = ["trade_date", "TRADE_DATE", "date", "日期", "交易日", "dt", "cal_date"]
    if not df.empty:
        lower = _lower_map(df)
        for c in candidates:
            key = str(c).strip().lower()
            if key in lower:
                v = str(df[lower[key]].iloc[0]).strip()
                v = v.replace("-", "").replace("/", "")
                if len(v) >= 8:
                    return v[:8]

    env_td = str(os.getenv("TRADE_DATE", "")).strip()
    env_td = env_td.replace("-", "").replace("/", "")
    if len(env_td) >= 8:
        return env_td[:8]

    return pd.Timestamp.today().strftime("%Y%m%d")


def _stable_jitter_from_ts(ts: str) -> float:
    """
    给排序做一个极小的确定性扰动，避免“全一样”导致肉眼误判。
    量级 1e-6，不会改变模型概率的量级。
    """
    s = str(ts or "").strip()
    if not s:
        return 0.0
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    v = int(h, 16) / float(16**8)  # 0~1
    return (v - 0.5) * 2e-6  # -1e-6 ~ +1e-6


def _write_feature_history(
    raw_input_df: pd.DataFrame,
    out_df: pd.DataFrame,
    trade_date: str,
    s=None,
) -> Dict[str, Any]:
    """
    写入 outputs/learning/feature_history.csv（追加模式）
    - 必须有 ts_code
    - 去重策略：同 trade_date + ts_code 去重，保留最新一条
    """
    try:
        raw_input_df = _normalize_id_columns(_ensure_df(raw_input_df))
        out_df = _ensure_df(out_df)

        if raw_input_df.empty:
            return {"ok": False, "reason": "empty raw_input"}

        if "ts_code" not in raw_input_df.columns:
            return {"ok": False, "reason": "missing ts_code in raw_input"}

        # 确保 out_df 有 FEATURES/Probability/_prob_src
        tmp = raw_input_df[["ts_code"]].copy()
        feat = _ensure_features(raw_input_df)
        for c in FEATURES:
            tmp[c] = feat[c].values if c in feat.columns else 0.0

        if "Probability" in out_df.columns:
            tmp["Probability"] = pd.to_numeric(out_df["Probability"], errors="coerce")
        else:
            tmp["Probability"] = np.nan

        tmp["_prob_src"] = out_df["_prob_src"].astype(str).fillna("") if "_prob_src" in out_df.columns else ""
        tmp["trade_date"] = str(trade_date)

        # 输出路径（仓库根目录 outputs/learning）
        base = Path("outputs") / "learning"
        _ensure_dir(base)
        fp = base / "feature_history.csv"

        if fp.exists():
            old = pd.read_csv(fp, dtype=str, encoding="utf-8")
            merged = pd.concat([old, tmp.astype(str)], ignore_index=True)
        else:
            merged = tmp.astype(str)

        merged["trade_date"] = merged["trade_date"].astype(str).str.strip()
        merged["ts_code"] = merged["ts_code"].astype(str).str.strip()
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

        # 排序：按 trade_date，再按 Probability
        try:
            merged["_prob_f"] = pd.to_numeric(merged.get("Probability", 0), errors="coerce").fillna(0.0)
            merged = merged.sort_values(["trade_date", "_prob_f"], ascending=[True, False]).drop(columns=["_prob_f"])
        except Exception:
            merged = merged.sort_values(["trade_date"], ascending=[True])

        merged.to_csv(fp, index=False, encoding="utf-8")
        return {"ok": True, "path": str(fp), "rows": int(len(merged)), "trade_date": str(trade_date)}
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
    """用 next_day 的 limit_list_d.csv 打标：在涨停列表里 => y=1"""
    ll = _read_csv_if_exists(next_snap / "limit_list_d.csv")
    if ll.empty:
        return set()

    ll = _normalize_id_columns(ll)
    if "ts_code" not in ll.columns:
        return set()

    codes = set()
    for v in ll["ts_code"].astype(str).tolist():
        v = str(v).strip()
        if not v:
            continue
        codes.add(v)
        codes.add(_to_nosuffix(v))
    return codes


def _build_xy_from_history(
    snapshot_dirs: List[Tuple[str, Path]],
    theme_file_name: str = "step4_theme.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    snapshot_dirs: [(trade_date, snap_path)]，按日期升序
    对每一天 d，用 d 的 step4_theme.csv 作为特征，用 d+1 的涨停列表打标
    """
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []

    for i in range(len(snapshot_dirs) - 1):
        _, snap = snapshot_dirs[i]
        _, snap_next = snapshot_dirs[i + 1]

        df_day = _read_csv_if_exists(snap / theme_file_name)
        if df_day.empty:
            continue

        df_day = _normalize_id_columns(df_day)
        if "ts_code" not in df_day.columns:
            continue

        codes = df_day["ts_code"].astype(str).str.strip().values
        feat = _ensure_features(df_day)
        if feat.empty:
            continue

        limit_set = _label_from_next_day_limit_list(snap_next)
        y = np.array(
            [1 if (c in limit_set or _to_nosuffix(c) in limit_set) else 0 for c in codes],
            dtype=int,
        )
        X = feat[FEATURES].astype(float).values

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
    """从 s.data_repo 获取最近 lookback+1 天快照目录（+1 用于 next_day 打标）"""
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
        root = getattr(dr, "warehouse_root", None)
        repo_name = getattr(dr, "repo_name", None)
        raw_dir = getattr(dr, "raw_dir", None)

        if root and repo_name and raw_dir:
            raw_root = Path(root) / repo_name / raw_dir
            if raw_root.exists():
                tmp = []
                for ydir in raw_root.iterdir():
                    if not ydir.is_dir():
                        continue
                    for ddir in ydir.iterdir():
                        if ddir.is_dir() and len(ddir.name) >= 8:
                            tmp.append(ddir.name)
                dates = sorted(tmp)
        else:
            base = Path("data_repo/snapshots")
            if base.exists():
                dates = sorted([p.name for p in base.iterdir() if p.is_dir()])

    dates = [d for d in dates if isinstance(d, str) and len(d) >= 8]
    dates = dates[-(lookback + 1):]

    out: List[Tuple[str, Path]] = []
    for d in dates:
        if hasattr(dr, "snapshot_dir"):
            try:
                p = Path(dr.snapshot_dir(d))
            except Exception:
                p = Path("data_repo/snapshots") / d
        else:
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
    """统一入口：每天收盘后先训练（尽量两种都训）"""
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

    ✅ 每次推断后自动写入 feature_history.csv
    ✅ 若模型输出“几乎全一样”，自动用 pseudo 做轻量 tie-break（解决你截图那种全同概率）
    """
    raw_input = _ensure_df(theme_df)
    if raw_input.empty:
        out = pd.DataFrame()
        out["Probability"] = pd.Series(dtype="float64")
        out["_prob_src"] = pd.Series(dtype="object")
        return out

    # 保留全字段（非常关键）
    out = raw_input.copy()

    # 统一 id + 特征
    ts_norm = _normalize_id_columns(raw_input)
    ts_series = (
        ts_norm["ts_code"].astype(str).str.strip()
        if "ts_code" in ts_norm.columns
        else pd.Series([""] * len(out))
    )

    feat = _ensure_features(raw_input)
    X = feat[FEATURES].astype(float).values

    # 伪概率（用于兜底，也用于“同分时”做 tie-break）
    def _pseudo_proba() -> np.ndarray:
        strength = np.clip(feat["StrengthScore"].astype(float).values / 100.0, 0.0, 1.5)
        theme = np.clip(feat["ThemeBoost"].astype(float).values, 0.0, 2.0)

        seal = feat["seal_amount"].astype(float).values
        seal = _safe_log1p_pos(seal)
        seal = np.clip(seal / 16.0, 0.0, 1.5)

        opens = np.clip(feat["open_times"].astype(float).values, 0.0, 20.0)
        turnover = np.clip(feat["turnover_rate"].astype(float).values, 0.0, 50.0) / 50.0

        z = (
            1.20 * strength
            + 1.10 * theme
            + 0.90 * seal
            + 0.35 * turnover
            - 0.60 * (opens / 10.0)
        )
        return _sigmoid(z)

    def _post_process(proba: np.ndarray, src: str) -> pd.DataFrame:
        # clip + 极小确定性 jitter（防止完全一样导致排序肉眼误判）
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba + np.array([_stable_jitter_from_ts(t) for t in ts_series.values], dtype=float)

        # 如果模型输出“几乎常数”，用 pseudo 做轻量 tie-break（非常关键）
        try:
            if np.nanstd(proba) < 1e-8:
                p2 = _pseudo_proba()
                proba = 0.98 * proba + 0.02 * p2
                src = f"{src}+tie"
        except Exception:
            pass

        out2 = out.copy()
        out2["Probability"] = np.clip(proba, 0.0, 1.0)
        out2["_prob_src"] = src

        td = _guess_trade_date(raw_input)
        _write_feature_history(raw_input_df=raw_input, out_df=out2, trade_date=td, s=s)
        return out2.sort_values("Probability", ascending=False)

    # 1) LGBM
    m_lgbm = load_lgbm(s=s)
    if m_lgbm is not None:
        try:
            proba = m_lgbm.predict_proba(X)[:, 1]
            return _post_process(proba, "lgbm")
        except Exception:
            pass

    # 2) LR
    m_lr = load_lr(s=s)
    if m_lr is not None:
        try:
            proba = m_lr.predict_proba(X)[:, 1]
            return _post_process(proba, "lr")
        except Exception:
            pass

    # 3) 兜底 pseudo
    return _post_process(_pseudo_proba(), "pseudo")


def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(df, s=s)


if __name__ == "__main__":
    print("Step5 Probability model loaded.")
