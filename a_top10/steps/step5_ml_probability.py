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
import json
from datetime import datetime
import re
mport hashlib
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
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from a_top10.config import Settings

# -------------------------
# Feature set
# -------------------------
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
    # 强制数值
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _guess_trade_date(df: pd.DataFrame) -> str:
    """
    尽量从 df 或环境变量推断 trade_date；若失败用当天日期。
    """
    df = _ensure_df(df)
    for c in ["trade_date", "TradeDate", "日期", "交易日期"]:
        if c in df.columns:
            v = str(df[c].iloc[0]).strip()
            if re.match(r"^\d{8}$", v):
                return v

    env_td = os.getenv("TRADE_DATE", "").strip()
    if re.match(r"^\d{8}$", env_td):
        return env_td

    # GitHub Actions 环境里通常没有“交易日”，这里取当天即可（后续 Step0/Step2 也会保证 trade_date 正确）
    return pd.Timestamp.today().strftime("%Y%m%d")


def _stable_jitter_from_ts(ts: str) -> float:
    """
    给排序做一个极小的确定性扰动，避免“全一样”导致肉眼误判。

    ⚠️ 修复点：
    - 原先 1e-6 量级在 writers 或报告显示保留 5 位小数时会被“截断成一样”
    - 这里提升到 1e-4 量级（仍然很小，不改变概率量级），但能在报告里看到差异
    """
    s = str(ts or "").strip()
    if not s:
        return 0.0
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    v = int(h, 16) / float(16**8)  # 0~1
    return (v - 0.5) * 2e-4  # -1e-4 ~ +1e-4


def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_sampling_state(outputs_dir: Path) -> Dict[str, Any]:
    """读取 Step7 生成的采样状态。不存在则返回默认值。"""
    p = outputs_dir / "learning" / "sampling_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_target_rows_per_day(s=None) -> int:
    """自动采样：默认 200，若 sampling_state.json 存在则按其 target_rows_per_day。"""
    outputs_dir = Path(getattr(getattr(s, "io", None), "outputs_dir", "outputs")) if s is not None else Path("outputs")
    st = _read_sampling_state(outputs_dir)
    try:
        v = int(st.get("target_rows_per_day", 200))
    except Exception:
        v = 200
    # 安全夹紧：防止写入过大导致仓库膨胀或 CI 超时
    return int(max(50, min(2000, v)))


def _write_feature_history(
    raw_input_df: pd.DataFrame,
    out_df: pd.DataFrame,
    trade_date: str,
    s=None,
) -> Dict[str, Any]:
    """
    写入 outputs/learning/feature_history.csv（追加模式）

    ✅ 自动采样（Auto-Sampling）：
    - 默认每天写入 Full Ranking 的前 N 行（N=200/500/1000 由 Step7 输出 sampling_state.json 决定）
    - 同 trade_date + ts_code 去重，保留最新一条
    - 仅截断“当日写入的那一批”，历史不动

    ✅ 兼容旧列：run_time_utc/name/_prob_warn 等都保持
    """
    try:
        raw_input_df = _normalize_id_columns(_ensure_df(raw_input_df))
        out_df = _ensure_df(out_df)

        if raw_input_df.empty:
            return {"ok": False, "reason": "empty raw_input"}

        if "ts_code" not in raw_input_df.columns:
            return {"ok": False, "reason": "missing ts_code in raw_input"}

        # 目标写入行数（由 Step7 采样状态控制）
        target_n = _get_target_rows_per_day(s=s)

        # 基础列
        tmp = pd.DataFrame()
        tmp["ts_code"] = raw_input_df["ts_code"].astype(str).str.strip()

        # name（若存在）
        name_col = None
        for c in ["name", "stock_name", "名称", "股票简称", "证券名称"]:
            if c in raw_input_df.columns:
                name_col = c
                break
        tmp["name"] = raw_input_df[name_col].astype(str) if name_col else ""

        # 特征列（确保 FEATURES 都有）
        feat = _ensure_features(raw_input_df)
        for c in FEATURES:
            if c in feat.columns:
                tmp[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
            else:
                tmp[c] = 0.0

        # 概率与来源
        tmp["Probability"] = pd.to_numeric(out_df.get("Probability", np.nan), errors="coerce")
        tmp["_prob_src"] = out_df.get("_prob_src", "").astype(str).fillna("") if "_prob_src" in out_df.columns else ""
        tmp["_prob_warn"] = out_df.get("_prob_warn", "").astype(str).fillna("") if "_prob_warn" in out_df.columns else ""

        # 时间列：必须写入（否则无法追溯）
        tmp["run_time_utc"] = _utc_now_iso()
        tmp["trade_date"] = str(trade_date).strip()

        # 只保留当日 TopN（Full Ranking by Probability）
        tmp["_prob_f"] = pd.to_numeric(tmp["Probability"], errors="coerce").fillna(0.0)
        tmp = tmp.sort_values("_prob_f", ascending=False).drop(columns=["_prob_f"])
        if target_n > 0:
            tmp = tmp.head(int(target_n))

        # 输出路径
        outputs_dir = Path(getattr(getattr(s, "io", None), "outputs_dir", "outputs")) if s is not None else Path("outputs")
        base = outputs_dir / "learning"
        _ensure_dir(base)
        fp = base / "feature_history.csv"

        # 读旧 + 合并
        if fp.exists():
            old = pd.read_csv(fp, dtype=str, encoding="utf-8")
            merged = pd.concat([old, tmp.astype(str)], ignore_index=True, sort=False)
        else:
            merged = tmp.astype(str)

        # 统一清洗 + 去重
        merged["trade_date"] = merged.get("trade_date", "").astype(str).str.strip()
        merged["ts_code"] = merged.get("ts_code", "").astype(str).str.strip()
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

        # 排序：trade_date 升序、Probability 降序
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
        pass


def load_lr(s=None):
    paths = _get_model_paths(s=s)
    return _load_joblib(paths.lr_path)


def load_lgbm(s=None):
    paths = _get_model_paths(s=s)
    return _load_joblib(paths.lgbm_path)


# -------------------------
# Train
# -------------------------
def _load_step4_theme_history(s, lookback: int, theme_file_name: str) -> pd.DataFrame:
    """
    从 outputs/learning/step4_theme.csv 读取历史 step4 输出（训练样本输入）
    """
    outputs_dir = Path(getattr(s.io, "outputs_dir", "outputs"))
    fp = outputs_dir / "learning" / theme_file_name
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        try:
            df = pd.read_csv(fp, encoding="gbk")
        except Exception:
            return pd.DataFrame()

    # 只保留最近 lookback 天（按 trade_date）
    if "trade_date" in df.columns:
        dts = sorted(df["trade_date"].astype(str).unique())
        use = set(dts[-lookback:]) if len(dts) > lookback else set(dts)
        df = df[df["trade_date"].astype(str).isin(use)].copy()
    return df


def _build_X_y_from_theme_history(s, lookback: int, theme_file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化训练：用 step4_theme.csv 的 FEATURES 做输入，用“是否涨停”做标签。
    标签来源：下一交易日 limit_list_d.csv
    """
    hist = _load_step4_theme_history(s, lookback=lookback, theme_file_name=theme_file_name)
    hist = _normalize_id_columns(hist)
    if hist.empty or "trade_date" not in hist.columns or "ts_code" not in hist.columns:
        return np.zeros((0, len(FEATURES))), np.zeros((0,))

    # 构造标签：对每个 trade_date 的样本，用 next_day limit_list 做 y
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

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
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
        # 先做数值清洗
        proba = np.asarray(proba, dtype=float)
        proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

        # clip + 确定性 jitter（防止完全一样导致排序肉眼误判）
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba + np.array([_stable_jitter_from_ts(t) for t in ts_series.values], dtype=float)

        # 如果模型输出“几乎常数”，用 pseudo 做更明显的 tie-break
        try:
            if np.nanstd(proba) < 1e-8:
                p2 = _pseudo_proba()
                # ✅ 修复点：增强 tie-break 权重，否则显示仍会“像全一样”
                proba = 0.80 * proba + 0.20 * p2
                src = f"{src}+tie"
        except Exception:
            pass

        # 再次 clip，保证合法
        proba = np.clip(proba, 0.0, 1.0)

        out2 = out.copy()
        out2["Probability"] = proba
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
