#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step7：自学习闭环更新模型权重（命中率滚动提升）

目标（终版）：
- 回收 outputs/predict_top10_*.md 的 TopN 预测
- 用 “下一交易日”的 limit_list_d.csv 打标（命中统计）
- 训练样本来源（强制主路径）：
    outputs/learning/feature_history.csv
- 训练输出：
    models/step5_lr.joblib
    models/step5_lgbm.joblib
- 输出学习报告：
    outputs/learning/step7_report_latest.json
    outputs/learning/step7_report_latest.md

✅ 本版本修复点（关键）：
- 方案A：自动寻找“下一个可用对照日”（跳过周末/节假日/缺快照日）
  -> expected_next_trade_date 可能不存在（如 20260216），自动滚动到第一个存在 limit_list_d 的实际交易日
- 移除 tushare 兜底（避免环境缺包带来噪音告警）
- 报告中同时披露 expected/actual next_trade_date，避免误读命中率

✅ 本次新增修复（你提的 Warnings 问题）：
- 若 expected_next_trade_date > today（未来对照日尚未发生）
  -> 直接判定为 PENDING（非错误）
  -> 不查 GitHub raw、不 forward 扫未来、不产生 404 warnings
"""

from __future__ import annotations

import json
import os
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings
from a_top10.steps.step5_ml_probability import _get_ts_code_col, _normalize_id_columns


# ============================================================
# Constants
# ============================================================
LR_MODEL_PATH = Path("step5_lr.joblib")
LGBM_MODEL_PATH = Path("step5_lgbm.joblib")

MIN_SAMPLES = 200  # 冷启动最低样本量
MIN_POS = 10       # 冷启动最低正样本数（涨停=1）

# 方案A：最多向后寻找多少天的对照日（周末/节假日/同步延迟）
MAX_FORWARD_LABEL_DAYS = int(os.getenv("MAX_FORWARD_LABEL_DAYS", "15") or "15")

# 用于判断“未来对照日未到”的时区（与主系统一致）
DEFAULT_TZ = os.getenv("A_TOP10_TZ", "Asia/Shanghai")


# ============================================================
# Utils
# ============================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _safe_write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_yyyymmdd() -> str:
    """
    ✅ 新增：统一“今天”的交易日判断基准（用于 future gate）
    - 默认用 Asia/Shanghai（与 workflow TZ 一致）
    """
    try:
        return pd.Timestamp.now(tz=DEFAULT_TZ).strftime("%Y%m%d")
    except Exception:
        # 兜底：用本地时间（不理想，但不会崩）
        return pd.Timestamp.now().strftime("%Y%m%d")


def _to_nosuffix(ts: str) -> str:
    ts = str(ts).strip()
    if not ts:
        return ts
    return ts.split(".")[0]


def _trade_date_from_output_path(p: Path) -> Optional[str]:
    m = re.match(r"predict_top10_(\d{8})\.md$", Path(p).name)
    return m.group(1) if m else None


def _read_text_best_effort(p: Path) -> str:
    """兜底读取：避免编码问题导致 Step7 崩溃"""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                return p.read_text(encoding="gbk", errors="ignore")
            except Exception:
                return ""


# ============================================================
# ✅ 从报告标题解析 “预测日 -> 目标日（下一交易日）”
# ============================================================

def _extract_pred_and_target_from_report(text: str) -> Tuple[str, str]:
    """
    尝试从报告标题中解析：
      《20260212 预测：20260213 涨停 TOP 10》
    返回：(pred_date, target_date)
    若找不到则返回 ("","")
    """
    if not text:
        return "", ""

    pat = re.compile(r"《\s*(\d{8})\s*预测：\s*(\d{8})\s*涨停\s*TOP\s*10\s*》")
    m = pat.search(text)
    if m:
        return m.group(1), m.group(2)

    pat2 = re.compile(r"《\s*(\d{8})\s*预测：\s*(\d{8})\s*涨停\s*TOP\s*10", re.IGNORECASE)
    m2 = pat2.search(text)
    if m2:
        return m2.group(1), m2.group(2)

    return "", ""


# ============================================================
# ✅ 只解析“涨停 TOP10”那张表的 TopN codes（支持 md 管道表 / html table）
# ============================================================

def _parse_topn_codes(md_path: Path, topn: int) -> List[str]:
    """
    从预测报告 md 中解析 TopN 代码：
    - 只解析“涨停 TOP 10”附近的第一张表
    - 支持：
        1) markdown pipe table
        2) html table
    """
    text = _read_text_best_effort(md_path)
    if not text:
        return []

    anchor = None
    for key in ["涨停 TOP 10", "涨停TOP 10", "涨停 TOP10", "涨停TOP10"]:
        idx = text.find(key)
        if idx >= 0:
            anchor = idx
            break

    scope = text[anchor:] if anchor is not None else text

    codes: List[str] = []
    m_table = re.search(r"<table\b.*?>.*?</table>", scope, flags=re.IGNORECASE | re.DOTALL)
    if m_table:
        table_html = m_table.group(0)
        for m in re.finditer(r"<tr>\s*<td>\s*\d+\s*</td>\s*<td>\s*([^<\s]+)\s*</td>", table_html, flags=re.IGNORECASE):
            codes.append(m.group(1).strip())
            if len(codes) >= int(topn):
                break
        return [c for c in codes if c][: int(topn)]

    lines = scope.splitlines()
    in_table = False
    for line in lines:
        s = line.strip()
        if not s:
            if in_table:
                break
            continue
        if s.startswith("|") and s.endswith("|"):
            in_table = True
            m = re.match(r"^\|\s*\d+\s*\|\s*([0-9A-Za-z\.\-]+)\s*\|", s)
            if m:
                codes.append(m.group(1).strip())
                if len(codes) >= int(topn):
                    break
        else:
            if in_table:
                break

    return [c for c in codes if c][: int(topn)]


def _limit_codes_from_df(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    df = _normalize_id_columns(df)
    ts_col = _get_ts_code_col(df)
    if ts_col is None or ts_col not in df.columns:
        return []
    vals: List[str] = []
    for v in df[ts_col].tolist():
        v = str(v).strip()
        if not v:
            continue
        vals.append(v)
        vals.append(_to_nosuffix(v))
    return vals


def _get_outputs_dir(s: Settings) -> Path:
    try:
        return Path(getattr(s.io, "outputs_dir", "outputs"))
    except Exception:
        return Path("outputs")


def _get_models_dir(s: Settings) -> Path:
    try:
        if hasattr(s, "data_repo") and hasattr(s.data_repo, "models_dir"):
            p = Path(getattr(s.data_repo, "models_dir"))
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass
    p = Path("models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_predict_files(outputs_dir: Path) -> List[Path]:
    return sorted(list(outputs_dir.glob("predict_top10_*.md")))


def _infer_next_trade_date(predict_dates: List[str], d: str) -> str:
    try:
        idx = predict_dates.index(d)
        if idx < len(predict_dates) - 1:
            return predict_dates[idx + 1]
    except Exception:
        pass
    return ""


# ============================================================
# Label source (关键修复)
# ============================================================

def _read_limit_list_from_github_raw(next_d: str, warnings: List[str]) -> pd.DataFrame:
    """
    兜底：从 GitHub raw 拉 limit_list_d.csv

    ✅ 固定读取当前仓库 a-top10 的 _warehouse 路径，避免 env 拼错导致 404
    """
    branch = str(os.getenv("DATA_BRANCH", "main")).strip()

    repo_full = str(os.getenv("GITHUB_REPOSITORY", "")).strip()
    if not repo_full:
        repo_full = "njedu2023-prog/a-top10"

    year = next_d[:4]
    url = (
        f"https://raw.githubusercontent.com/{repo_full}/{branch}"
        f"/_warehouse/a-share-top3-data/data/raw/{year}/{next_d}/limit_list_d.csv"
    )
    warnings.append(f"github_raw_try: {url}")

    try:
        return pd.read_csv(url, dtype=str, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(url, dtype=str, encoding="utf-8")
        except Exception as e:
            warnings.append(f"github_raw read limit_list_d failed: {e}")
            return pd.DataFrame()


def _resolve_next_trade_date_by_snapshot(
    s: Settings,
    expected_next_d: str,
    warnings: List[str],
    max_forward_days: int = MAX_FORWARD_LABEL_DAYS,
) -> Tuple[str, str]:
    """
    方案A：自动寻找“下一个可用对照日”
    - expected_next_d 可能是非交易日/还未同步
    - 返回 (actual_next_d, src)
      - 找到则 actual_next_d=YYYYMMDD
      - 找不到则 actual_next_d=""
    ✅ 修复：若 expected_next_d 在未来（> today），直接返回 PENDING，不查 raw、不 forward 扫未来
    ✅ 修复：forward 搜索上界限制到 today（只在“已到期但缺文件”时才 forward）
    """
    if not expected_next_d or not re.match(r"^\d{8}$", str(expected_next_d)):
        return "", "bad_expected_next_d"

    today = _today_yyyymmdd()
    if str(expected_next_d) > str(today):
        # ✅ future gate：未来对照未发生 -> 正常 pending，不应产生 404 warnings
        return "", "pending_future"

    d0 = pd.to_datetime(expected_next_d, format="%Y%m%d", errors="coerce")
    if pd.isna(d0):
        return "", "bad_expected_next_d"

    # ✅ 上界：最多只能扫到 today（避免扫进未来）
    try:
        d_today = pd.to_datetime(today, format="%Y%m%d", errors="coerce")
        max_days_until_today = int((d_today - d0).days) if (not pd.isna(d_today)) else int(max_forward_days)
        max_i = min(int(max_forward_days), max(0, max_days_until_today))
    except Exception:
        max_i = int(max_forward_days)

    for i in range(0, int(max_i) + 1):
        di = (d0 + pd.Timedelta(days=i)).strftime("%Y%m%d")

        # 1) warehouse
        try:
            dfw = s.data_repo.read_limit_list(di)  # type: ignore[attr-defined]
            if dfw is not None and not dfw.empty:
                if i > 0:
                    warnings.append(f"next_trade_date_adjusted: {expected_next_d} -> {di} (warehouse)")
                return di, "warehouse"
        except Exception:
            pass

        # 2) github raw（只会在 di <= today 的范围内触发）
        dfr = _read_limit_list_from_github_raw(di, warnings)
        if dfr is not None and not dfr.empty:
            if i > 0:
                warnings.append(f"next_trade_date_adjusted: {expected_next_d} -> {di} (github_raw)")
            return di, "github_raw"

    warnings.append(f"next_trade_date_unresolved: expected={expected_next_d} max_forward_days={max_forward_days} upper_bound={today}")
    return "", "not_found_within_window"


def _read_limit_list_anyway(s: Settings, expected_trade_date: str, warnings: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    ✅ 关键：先把 expected_trade_date resolve 成“真实可用对照日”，再读取 limit_list_d
    返回：(df, actual_trade_date)
    """
    actual_d, src = _resolve_next_trade_date_by_snapshot(
        s=s,
        expected_next_d=expected_trade_date,
        warnings=warnings,
        max_forward_days=MAX_FORWARD_LABEL_DAYS,
    )
    if not actual_d:
        # pending_future 也在这里体现为 “无对照可读”
        return pd.DataFrame(), ""

    # 1) warehouse（再次读取，避免重复 raw）
    try:
        df = s.data_repo.read_limit_list(actual_d)  # type: ignore[attr-defined]
        if df is not None and not df.empty:
            return df, actual_d
    except Exception as e:
        warnings.append(f"read_limit_list (warehouse) failed: {e}")

    # 2) github raw
    df2 = _read_limit_list_from_github_raw(actual_d, warnings)
    return (df2 if df2 is not None else pd.DataFrame()), actual_d


# ============================================================
# Build train set from feature_history
# ============================================================

def _build_train_set_from_feature_history(
    s: Settings,
    lookback_days: int,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    训练样本：
    - X：StrengthScore/ThemeBoost/seal_amount/open_times/turnover_rate
    - y：next_day 是否涨停（用“实际可用对照日”的 limit_list_d.csv 打标）
    """
    outputs_dir = _get_outputs_dir(s)
    fp = outputs_dir / "learning" / "feature_history.csv"

    meta: Dict[str, Any] = {
        "feature_history_file": str(fp),
        "rows_raw": 0,
        "rows_used": 0,
        "rows_dropped_allzero": 0,
        "dates_total": 0,
        "dates_used": 0,
    }

    if not fp.exists():
        warnings.append("feature_history.csv not found: training skipped.")
        return pd.DataFrame(), meta

    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(fp, dtype=str, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        try:
            df = pd.read_csv(fp, dtype=str)
        except Exception as e:
            warnings.append(f"read feature_history failed: {e}")
            return pd.DataFrame(), meta

    if df is None or df.empty:
        warnings.append("feature_history.csv empty: training skipped.")
        return pd.DataFrame(), meta

    meta["rows_raw"] = int(len(df))

    for c in ["trade_date", "ts_code"]:
        if c not in df.columns:
            warnings.append(f"feature_history missing col: {c}")
            return pd.DataFrame(), meta

    df["trade_date"] = df["trade_date"].astype(str).str.strip()
    df["ts_code"] = df["ts_code"].astype(str).str.strip()

    dates = sorted([d for d in df["trade_date"].unique().tolist() if re.match(r"^\d{8}$", str(d))])
    meta["dates_total"] = int(len(dates))
    use_dates = dates[-lookback_days:] if len(dates) > lookback_days else dates
    meta["dates_used"] = int(len(use_dates))

    dfx = df[df["trade_date"].isin(use_dates)].copy()
    if dfx.empty:
        warnings.append("feature_history filtered empty by lookback.")
        return pd.DataFrame(), meta

    feats = ["StrengthScore", "ThemeBoost", "seal_amount", "open_times", "turnover_rate"]
    for c in feats:
        if c not in dfx.columns:
            dfx[c] = "0"
    for c in feats:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce").fillna(0.0)

    allzero = (dfx[feats].abs().sum(axis=1) <= 0.0)
    meta["rows_dropped_allzero"] = int(allzero.sum())
    dfx = dfx[~allzero].copy()
    if dfx.empty:
        warnings.append("all rows are all-zero features: training skipped.")
        return pd.DataFrame(), meta

    dates2 = sorted([d for d in dfx["trade_date"].unique().tolist() if re.match(r"^\d{8}$", str(d))])
    rows: List[Dict[str, Any]] = []

    for i, d in enumerate(dates2[:-1]):
        expected_next_d = dates2[i + 1]

        lim_df, actual_next_d = _read_limit_list_anyway(s, expected_next_d, warnings)
        lim_set = set(_limit_codes_from_df(lim_df))

        if not actual_next_d:
            # 这里多数是“未同步”或“缺快照”，但也可能是 pending_future（极少）
            warnings.append(f"label_pending: trade_date={d} expected_next={expected_next_d} (no snapshot yet)")
            continue

        if not lim_set:
            warnings.append(f"label_source_empty: trade_date={d} actual_next_trade_date={actual_next_d} limit_list empty")

        df_day = dfx[dfx["trade_date"] == d].copy()
        if df_day.empty:
            continue

        for _, r in df_day.iterrows():
            code = str(r.get("ts_code", "")).strip()
            y = 1 if (code in lim_set or _to_nosuffix(code) in lim_set) else 0
            row = {
                "trade_date": d,
                "next_trade_date": actual_next_d,          # ✅ 记录实际 used 对照日
                "expected_next_trade_date": expected_next_d,
                "ts_code": code,
                "label": int(y),
            }
            for c in feats:
                row[c] = float(r.get(c, 0.0))
            rows.append(row)

    train_df = pd.DataFrame(rows)
    meta["rows_used"] = int(len(train_df))
    return train_df, meta


# ============================================================
# Train models and save
# ============================================================

def _train_lr(train_df: pd.DataFrame, warnings: List[str]):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    feats = ["StrengthScore", "ThemeBoost", "seal_amount", "open_times", "turnover_rate"]
    X = train_df[feats].astype(float).values
    y = train_df["label"].astype(int).values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model


def _train_lgbm(train_df: pd.DataFrame, warnings: List[str]):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        warnings.append("lightgbm not installed: skip lgbm.")
        return None

    feats = ["StrengthScore", "ThemeBoost", "seal_amount", "open_times", "turnover_rate"]
    X = train_df[feats].astype(float).values
    y = train_df["label"].astype(int).values

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _save_joblib_model(model, path: Path, warnings: List[str]) -> bool:
    if model is None:
        return False
    try:
        import joblib
    except Exception:
        warnings.append("joblib not installed: cannot save model.")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return True
    except Exception as e:
        warnings.append(f"save model failed: {e}")
        return False


def _train_and_save_models(s: Settings, train_df: pd.DataFrame, warnings: List[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "trained": False,
        "lr_saved": False,
        "lgbm_saved": False,
        "models_dir": "",
        "detail": {},
    }

    if train_df is None or train_df.empty:
        warnings.append("train_df empty: training skipped.")
        res["detail"] = {"reason": "train_df empty", "train_rows": 0, "pos": 0, "neg": 0}
        return res

    pos = int(train_df["label"].astype(int).sum())
    neg = int(len(train_df) - pos)
    n = int(len(train_df))

    res["detail"]["train_rows"] = n
    res["detail"]["pos"] = pos
    res["detail"]["neg"] = neg

    if n < MIN_SAMPLES:
        warnings.append(f"not enough samples for cold start: n={n} < {MIN_SAMPLES}")
        res["detail"]["reason"] = "min_samples"
        return res
    if pos < MIN_POS:
        warnings.append(f"not enough positive labels for cold start: pos={pos} < {MIN_POS}")
        res["detail"]["reason"] = "min_pos"
        return res
    if pos == 0 or neg == 0:
        warnings.append("only one class present (all 0 or all 1): training skipped.")
        res["detail"]["reason"] = "single_class"
        return res

    models_dir = _get_models_dir(s)
    res["models_dir"] = str(models_dir)

    lr_model = _train_lr(train_df, warnings)
    lr_path = models_dir / LR_MODEL_PATH.name
    lr_saved = _save_joblib_model(lr_model, lr_path, warnings)
    res["lr_saved"] = bool(lr_saved)

    lgbm_model = _train_lgbm(train_df, warnings)
    lgbm_path = models_dir / LGBM_MODEL_PATH.name
    lgbm_saved = _save_joblib_model(lgbm_model, lgbm_path, warnings)
    res["lgbm_saved"] = bool(lgbm_saved)

    res["trained"] = bool(lr_saved or lgbm_saved)
    res["detail"]["lr_path"] = str(lr_path) if lr_saved else ""
    res["detail"]["lgbm_path"] = str(lgbm_path) if lgbm_saved else ""
    res["detail"]["reason"] = "ok" if res["trained"] else "save_failed"
    return res


# ============================================================
# Auto-Sampling + Quality Gate v1 (Contract 固化)
# ============================================================

SAMPLING_STATE_FILE = "sampling_state.json"

CORE_FEATURE_COLS_V1 = [
    "StrengthScore",
    "ThemeBoost",
    "turnover_rate",
    "seal_amount",
    "open_times",
    "Probability",
]

META_COLS_V1 = ["trade_date", "ts_code", "_prob_src"]


def _read_feature_history(outputs_dir: Path) -> Tuple[pd.DataFrame, str]:
    fp = outputs_dir / "learning" / "feature_history.csv"
    if not fp.exists():
        return pd.DataFrame(), str(fp)
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(fp, dtype=str, encoding=enc), str(fp)
        except Exception:
            continue
    try:
        return pd.read_csv(fp, dtype=str), str(fp)
    except Exception:
        return pd.DataFrame(), str(fp)


def _rows_per_day_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "trade_date" not in df.columns:
        return pd.Series(dtype="int64")
    d = df.copy()
    d["trade_date"] = d["trade_date"].astype(str).str.strip()
    if "ts_code" not in d.columns:
        d["ts_code"] = ""
    return d.groupby("trade_date")["ts_code"].count().sort_index()


def _quality_gate_v1(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {"core_cols": CORE_FEATURE_COLS_V1, "meta_cols": META_COLS_V1}
    if df is None or df.empty:
        details["reason"] = "feature_history empty"
        details["pass"] = False
        return False, details

    missing = [c for c in (META_COLS_V1 + CORE_FEATURE_COLS_V1) if c not in df.columns]
    details["missing_cols"] = missing
    if missing:
        details["reason"] = "missing required cols"
        details["pass"] = False
        return False, details

    try:
        dates = sorted(df["trade_date"].astype(str).unique())
        recent_dates = dates[-30:] if len(dates) >= 30 else dates
        d = df[df["trade_date"].astype(str).isin(recent_dates)].copy()
    except Exception:
        d = df.copy()

    if d.empty:
        details["reason"] = "window empty"
        details["pass"] = False
        return False, details

    metrics: Dict[str, Any] = {}
    for c in CORE_FEATURE_COLS_V1:
        x = pd.to_numeric(d[c], errors="coerce")
        x0 = x.fillna(0.0)
        metrics[c] = {
            "non_null_rate": float(x.notna().mean()) if len(x) else 0.0,
            "non_zero_rate": float((x0 != 0.0).mean()) if len(x0) else 0.0,
            "std": float(x0.std()) if len(x0) else 0.0,
            "unique_count": int(x.dropna().nunique()),
        }

    try:
        src = d["_prob_src"].astype(str).str.lower().fillna("")
        pseudo_ratio = float((src == "pseudo").mean())
    except Exception:
        pseudo_ratio = 1.0

    details["metrics"] = metrics
    details["pseudo_ratio"] = pseudo_ratio
    details["window_days"] = int(len(sorted(d["trade_date"].astype(str).unique()))) if "trade_date" in d.columns else 0
    details["rows_in_window"] = int(len(d))

    ok = True
    for c in CORE_FEATURE_COLS_V1:
        if metrics[c]["non_null_rate"] < 0.90:
            ok = False
    for c in ["StrengthScore", "ThemeBoost", "Probability"]:
        if metrics.get(c, {}).get("std", 0.0) <= 0.0:
            ok = False
    if metrics.get("StrengthScore", {}).get("non_zero_rate", 0.0) < 0.30:
        ok = False
    if metrics.get("turnover_rate", {}).get("non_zero_rate", 0.0) < 0.30:
        ok = False
    if metrics.get("ThemeBoost", {}).get("unique_count", 0) < 6:
        ok = False
    if metrics.get("seal_amount", {}).get("non_zero_rate", 0.0) < 0.05:
        ok = False
    if (metrics.get("open_times", {}).get("std", 0.0) <= 0.0) and (metrics.get("open_times", {}).get("non_zero_rate", 0.0) < 0.02):
        ok = False

    details["pass"] = bool(ok)
    return bool(ok), details


def _count_ge(window: List[int], th: int) -> int:
    return int(sum(1 for v in window if v >= th))


def _decide_sampling_stage(
    prev_stage: str,
    days_covered: int,
    rows_last: List[int],
    quality_pass: bool,
) -> Tuple[str, int, Dict[str, Any]]:
    prev = (prev_stage or "S1_MVP").strip()
    rows = [int(x) for x in (rows_last or []) if x is not None]
    debug: Dict[str, Any] = {"prev_stage": prev, "days_covered": int(days_covered), "rows_last": rows[-20:]}

    stage = prev if prev else "S1_MVP"
    target = 200 if stage.startswith("S1") else 500 if stage.startswith("S2") else 1000

    if not quality_pass:
        debug["reason"] = "quality_gate_fail"
        return stage, target, debug

    if stage.startswith("S1"):
        w = rows[-10:]
        if days_covered >= 30 and _count_ge(w, 200) >= 7:
            return "S2_STD", 500, {**debug, "upgrade": "S1->S2"}

    if stage.startswith("S2"):
        w = rows[-15:]
        if days_covered >= 90 and _count_ge(w, 500) >= 10:
            return "S3_STRONG", 1000, {**debug, "upgrade": "S2->S3"}

    if stage.startswith("S3"):
        w = rows[-20:]
        debug["keep_check_ge_1000"] = _count_ge(w, 1000)
        return "S3_STRONG", 1000, debug

    return stage, target, debug


def _write_sampling_state(outputs_dir: Path, obj: Dict[str, Any]) -> str:
    p = outputs_dir / "learning" / SAMPLING_STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# ============================================================
# Main Step7
# ============================================================

def run_step7(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []

    outputs_dir = _get_outputs_dir(s)
    learning_dir = outputs_dir / "learning"
    _ensure_dir(outputs_dir)
    _ensure_dir(learning_dir)

    today = _today_yyyymmdd()  # ✅ 新增：统一 today

    try:
        topn = int(getattr(s, "topn", 10) or 10)
    except Exception:
        topn = 10

    try:
        lookback_days = int(getattr(s, "step7_lookback_days", getattr(s, "lookback_days", 150)) or 150)
    except Exception:
        lookback_days = 150

    # ---------------------------
    # 0) Auto-Sampling + Quality Gate
    # ---------------------------
    fh_df, fh_path = _read_feature_history(outputs_dir)
    rows_ser = _rows_per_day_series(fh_df)
    days_covered = int(len(rows_ser)) if len(rows_ser) else 0
    rows_last = [int(v) for v in rows_ser.tail(120).tolist()] if len(rows_ser) else []
    quality_pass, q_details = _quality_gate_v1(fh_df)

    prev_state: Dict[str, Any] = {}
    prev_p = learning_dir / SAMPLING_STATE_FILE
    if prev_p.exists():
        try:
            prev_state = json.loads(prev_p.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}

    prev_stage = str(prev_state.get("sampling_stage", "S1_MVP"))
    stage, target_rows, stage_debug = _decide_sampling_stage(
        prev_stage=prev_stage,
        days_covered=days_covered,
        rows_last=rows_last,
        quality_pass=bool(quality_pass),
    )

    sampling_state = {
        "sampling_stage": stage,
        "target_rows_per_day": int(target_rows),
        "days_covered": int(days_covered),
        "rows_per_day_last_N": rows_last[-120:],
        "quality_gate_pass": bool(quality_pass),
        "quality_gate_details": q_details,
        "stage_debug": stage_debug,
        "pseudo_ratio": float(q_details.get("pseudo_ratio", 1.0)),
        "feature_history_file": fh_path,
        "updated_at_utc": _utc_now_iso(),
        "model_version": str(os.getenv("GITHUB_SHA") or os.getenv("GITHUB_RUN_ID") or ""),
        "today_yyyymmdd": today,  # ✅ 记录 today
    }

    sampling_state_path = _write_sampling_state(outputs_dir, sampling_state)

    # ---------------------------
    # 1) 命中率统计（方案A增强：expected vs actual）
    # ---------------------------
    predict_files = _list_predict_files(outputs_dir)
    predict_dates = [d for d in [_trade_date_from_output_path(p) for p in predict_files] if d]
    predict_dates = sorted(list(dict.fromkeys(predict_dates)))

    hit_rows: List[Dict[str, Any]] = []
    date_to_file: Dict[str, Path] = {}
    for p in predict_files:
        d = _trade_date_from_output_path(p)
        if d:
            date_to_file[d] = p

    for d in predict_dates:
        md_path = date_to_file.get(d)
        if md_path is None or not md_path.exists():
            continue

        text = _read_text_best_effort(md_path)

        pred_in_title, target_in_title = _extract_pred_and_target_from_report(text)

        trade_date = pred_in_title or d
        expected_nd = target_in_title

        if not expected_nd:
            expected_nd = _infer_next_trade_date(predict_dates, d)
            if expected_nd:
                warnings.append(f"hit_rate_fallback_next_date: trade_date={trade_date} use_next={expected_nd} (title_parse_failed)")
            else:
                warnings.append(f"hit_rate_no_next_date: trade_date={trade_date} (title_parse_failed and no next file)")

        codes = _parse_topn_codes(md_path, topn=topn)

        if (not expected_nd) or (not codes):
            hit_rows.append({
                "trade_date": trade_date,
                "expected_next_trade_date": expected_nd or "",
                "actual_next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": ("no expected_next_trade_date" if not expected_nd else "no topn codes parsed"),
            })
            continue

        # ✅ 关键修复：未来对照日未到 -> 直接 pending，不查 raw、不 forward
        if str(expected_nd) > str(today):
            hit_rows.append({
                "trade_date": trade_date,
                "expected_next_trade_date": expected_nd,
                "actual_next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": f"pending_label: expected_next_trade_date_not_arrived (today={today})",
            })
            continue

        lim_df, actual_nd = _read_limit_list_anyway(s, expected_nd, warnings)

        if not actual_nd:
            # 对照数据未产生/未同步：不要写 0，写 pending
            hit_rows.append({
                "trade_date": trade_date,
                "expected_next_trade_date": expected_nd,
                "actual_next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": "pending_label: next trading day snapshot not ready",
            })
            continue

        lim_set = _limit_codes_from_df(lim_df)

        hit = 0
        for c in codes:
            if (c in lim_set) or (_to_nosuffix(c) in lim_set):
                hit += 1

        hit_rate = hit / max(1, len(codes))
        hit_rows.append({
            "trade_date": trade_date,
            "expected_next_trade_date": expected_nd,
            "actual_next_trade_date": actual_nd,
            "topn": len(codes),
            "hit": hit,
            "hit_rate": round(hit_rate, 4),
            "note": "",
        })

    hit_df = pd.DataFrame(hit_rows)
    hit_csv = learning_dir / "step7_hit_rate_history.csv"
    if not hit_df.empty:
        hit_df.to_csv(hit_csv, index=False, encoding="utf-8-sig")

    # ---------------------------
    # 2) 构建训练集 + 训练落盘模型
    # ---------------------------
    train_df, train_meta = _build_train_set_from_feature_history(
        s=s,
        lookback_days=lookback_days,
        warnings=warnings,
    )

    if bool(sampling_state.get("quality_gate_pass")):
        train_result = _train_and_save_models(s=s, train_df=train_df, warnings=warnings)
    else:
        if train_df is None or train_df.empty:
            tr = {"train_rows": 0, "pos": 0, "neg": 0}
        else:
            pos = int(train_df["label"].astype(int).sum())
            neg = int(len(train_df) - pos)
            tr = {"train_rows": int(len(train_df)), "pos": pos, "neg": neg}

        train_result = {
            "trained": False,
            "lr_saved": False,
            "lgbm_saved": False,
            "models_dir": "",
            "detail": {
                **tr,
                "reason": "skip_train: quality_gate_fail",
                "quality_gate_pass": bool(sampling_state.get("quality_gate_pass")),
                "days_covered": int(sampling_state.get("days_covered", 0)),
            },
        }
        warnings.append("skip_train: quality_gate_fail")

    # ---------------------------
    # 3) 输出学习报告
    # ---------------------------
    latest_hit = None
    if not hit_df.empty:
        # 只取“已完成对照”的最新一条（hit 不为空）
        v = hit_df[hit_df["hit"].astype(str).str.len() > 0]
        if not v.empty:
            latest_hit = v.iloc[-1].to_dict()

    report = {
        "ts": _now_str(),
        "topn": topn,
        "lookback_days": lookback_days,
        "today_yyyymmdd": today,
        "latest_hit": latest_hit,
        "train_meta": train_meta,
        "sampling_state": sampling_state,
        "sampling_state_path": sampling_state_path,
        "train_result": train_result,
        "warnings": warnings,
    }

    report_json = learning_dir / "step7_report_latest.json"
    _safe_write_json(report_json, report)

    md_lines: List[str] = []
    md_lines.append("# Step7 自学习报告（latest）")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{report['ts']}")
    md_lines.append(f"- Today：{today}")
    md_lines.append(f"- TopN：{topn}")
    md_lines.append(f"- Lookback：{lookback_days} 天")
    md_lines.append("")

    md_lines.append("## 1) 最新命中")
    if latest_hit:
        md_lines.append("")
        md_lines.append(f"- trade_date：{latest_hit.get('trade_date','')}")
        if "expected_next_trade_date" in latest_hit or "actual_next_trade_date" in latest_hit:
            md_lines.append(f"- expected_next_trade_date：{latest_hit.get('expected_next_trade_date','')}")
            md_lines.append(f"- actual_next_trade_date：{latest_hit.get('actual_next_trade_date','')}")
        else:
            md_lines.append(f"- next_trade_date：{latest_hit.get('next_trade_date','')}")
        md_lines.append(f"- hit/topn：{latest_hit.get('hit','')}/{latest_hit.get('topn','')}")
        md_lines.append(f"- hit_rate：{latest_hit.get('hit_rate','')}")
    else:
        md_lines.append("")
        md_lines.append("- 暂无可验证命中（对照日快照尚未产生/同步，或缺少下一交易日预测文件）")

    md_lines.append("")
    md_lines.append("## 1.5) Auto-Sampling 状态")
    md_lines.append("")
    md_lines.append(f"- sampling_stage：{sampling_state.get('sampling_stage','')}")
    md_lines.append(f"- target_rows_per_day：{sampling_state.get('target_rows_per_day','')}")
    md_lines.append(f"- days_covered：{sampling_state.get('days_covered','')}")
    md_lines.append(f"- quality_gate_pass：{sampling_state.get('quality_gate_pass','')}")
    md_lines.append(f"- pseudo_ratio：{sampling_state.get('pseudo_ratio','')}")
    md_lines.append("")

    md_lines.append("## 2) 训练数据概况")
    md_lines.append("")
    md_lines.append(f"- 特征历史文件：{train_meta.get('feature_history_file','') or '未找到'}")
    md_lines.append(f"- 原始行数：{train_meta.get('rows_raw',0)}")
    md_lines.append(f"- 过滤后行数：{train_meta.get('rows_used',0)}")
    md_lines.append(f"- 丢弃全零特征行：{train_meta.get('rows_dropped_allzero',0)}")
    md_lines.append(f"- 日期总数：{train_meta.get('dates_total',0)}")
    md_lines.append(f"- 使用日期：{train_meta.get('dates_used',0)}")

    md_lines.append("")
    md_lines.append("## 3) 训练执行结果")
    md_lines.append("")
    md_lines.append(f"- trained：{train_result.get('trained')}")
    md_lines.append(f"- lr_saved：{train_result.get('lr_saved')}")
    md_lines.append(f"- lgbm_saved：{train_result.get('lgbm_saved')}")
    md_lines.append(f"- models_dir：{train_result.get('models_dir','')}")
    if isinstance(train_result.get("detail"), dict):
        d = train_result["detail"]
        md_lines.append(f"- train_rows：{d.get('train_rows','')}")
        md_lines.append(f"- pos/neg：{d.get('pos','')}/{d.get('neg','')}")
        if d.get("reason"):
            md_lines.append(f"- reason：{d.get('reason')}")
        if d.get("lr_path"):
            md_lines.append(f"- lr_path：{d.get('lr_path')}")
        if d.get("lgbm_path"):
            md_lines.append(f"- lgbm_path：{d.get('lgbm_path')}")

    if warnings:
        md_lines.append("")
        md_lines.append("## 4) Warnings")
        md_lines.append("")
        for w in warnings[:80]:
            md_lines.append(f"- {w}")
        if len(warnings) > 80:
            md_lines.append(f"- ...（共 {len(warnings)} 条，仅展示前 80 条）")

    report_md = learning_dir / "step7_report_latest.md"
    _safe_write_text(report_md, "\n".join(md_lines) + "\n")

    return {
        "step7_learning": {
            "hit_history_csv": str(hit_csv),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "models_dir": str(train_result.get("models_dir", "")),
            "trained": bool(train_result.get("trained")),
            "train_rows": int(train_result.get("detail", {}).get("train_rows", 0)) if isinstance(train_result.get("detail"), dict) else 0,
            "warnings": warnings,
        }
    }


if __name__ == "__main__":
    s = Settings()
    out = run_step7(s, ctx={})
    print(json.dumps(out, ensure_ascii=False, indent=2))
