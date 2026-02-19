#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step7：自学习闭环更新模型权重（命中率滚动提升）

目标（终版）：
- 回收 outputs/predict_top10_*.md 的 TopN 预测
- 用 “下一交易日”的 limit_list_d.csv 打标（命中统计）
- 训练样本来源（强制主路径）：
    outputs/learning/feature_history.csv   （Step5 推断后自动追加）
  说明：这是唯一可靠的“跨日、可训练”样本来源。
- 训练输出：
    models/step5_lr.joblib
    models/step5_lgbm.joblib（可选，lightgbm 存在时）
- 输出学习报告：
    outputs/learning/step7_hit_rate_history.csv
    outputs/learning/step7_report_latest.json
    outputs/learning/step7_report_latest.md

注意（工程约束）：
- Step7 不能损坏主程序：任何缺文件/字段/库不匹配都必须“降级不报错”，只写 warnings。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings

# 仅复用 id 归一化，避免重复造轮子
from a_top10.steps.step5_ml_probability import _normalize_id_columns

# 训练依赖（best-effort）
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover
    LogisticRegression = None
    StandardScaler = None
    Pipeline = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


# ============================================================
# Constants
# ============================================================

FEATURES = ["StrengthScore", "ThemeBoost", "seal_amount", "open_times", "turnover_rate"]

FEATURE_ALIASES: Dict[str, List[str]] = {
    "StrengthScore": ["StrengthScore", "strengthscore", "strength_score", "强度得分", "强度", "强度分"],
    "ThemeBoost": ["ThemeBoost", "themeboost", "theme_boost", "题材加成", "题材"],
    "seal_amount": ["seal_amount", "sealamount", "seal_amt", "seal", "封单金额", "封单"],
    "open_times": ["open_times", "opentimes", "open_time", "open_count", "openings", "开板次数", "打开次数"],
    "turnover_rate": ["turnover_rate", "turnoverrate", "turn_rate", "turnover", "换手率", "换手率%"],
}

# feature_history 主文件（你仓库当前已存在）
FEATURE_HISTORY_PATH = Path("outputs") / "learning" / "feature_history.csv"

# 模型文件固定路径（与 Step5 对齐）
MODELS_DIR_DEFAULT = Path("models")
LR_MODEL_PATH = MODELS_DIR_DEFAULT / "step5_lr.joblib"
LGBM_MODEL_PATH = MODELS_DIR_DEFAULT / "step5_lgbm.joblib"

# 冷启动阈值（你当前数据量不大 + 正样本稀少，必须降门槛才能先落盘）
MIN_SAMPLES = 80
MIN_POS = 5

# ============================================================
# Utils
# ============================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_write_text(p: Path, text: str) -> None:
    _ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def _safe_write_json(p: Path, obj: Any) -> None:
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _to_nosuffix(ts: str) -> str:
    ts = str(ts).strip()
    if not ts:
        return ts
    return ts.split(".")[0]


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "<na>"):
        return ""
    return s


def _read_csv_guess(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, engine="python")
        except Exception:
            continue
    try:
        return pd.read_csv(path, dtype=str, engine="python")
    except Exception:
        return pd.DataFrame()


def _get_outputs_dir(s: Settings) -> Path:
    # 兼容 Settings.io.outputs_dir
    io = getattr(s, "io", None)
    if io is not None and getattr(io, "outputs_dir", None):
        try:
            return Path(getattr(io, "outputs_dir"))
        except Exception:
            pass
    return Path("outputs")


def _get_models_dir(s: Settings) -> Path:
    # 兼容 Settings.data_repo.models_dir / model_dir
    dr = getattr(s, "data_repo", None)
    if dr is not None:
        for attr in ("models_dir", "model_dir"):
            v = getattr(dr, attr, None)
            if v:
                try:
                    return Path(v)
                except Exception:
                    pass
        # 兼容 data_repo.root/models
        root = getattr(dr, "root", None)
        if root:
            try:
                return Path(root) / "models"
            except Exception:
                pass
    return MODELS_DIR_DEFAULT


# ============================================================
# Hit-rate (from predict_top10_*.md)
# ============================================================

def _trade_date_from_output_path(p: Path) -> Optional[str]:
    name = p.name
    m = re.match(r"predict_top10_(\d{8})(?:_.*)?\.md$", name)
    return m.group(1) if m else None


def _list_predict_files(outputs_dir: Path) -> List[Path]:
    if not outputs_dir.exists():
        return []
    ps = sorted(outputs_dir.glob("predict_top10_*.md"))
    return [p for p in ps if p.is_file()]


def _parse_topn_codes(md_path: Path, topn: int) -> List[str]:
    """
    解析 md 表格中 TopN 的股票代码（宽松）
    典型行：| 1 | 600000.SH | xxx | ...
    """
    text = _safe_read_text(md_path)
    if not text:
        return []

    codes: List[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*\|\s*\d+\s*\|\s*([0-9A-Za-z\.\-_]+)\s*\|", line)
        if m:
            codes.append(m.group(1).strip())

    codes = [c for c in codes if c]
    return codes[: int(topn)]


def _infer_next_trade_date(trade_dates_sorted: List[str], cur_date: str) -> Optional[str]:
    try:
        idx = trade_dates_sorted.index(cur_date)
    except ValueError:
        return None
    if idx + 1 >= len(trade_dates_sorted):
        return None
    return trade_dates_sorted[idx + 1]


# ============================================================
# Limit list reading (robust)
# ============================================================

def _limit_codes_from_df(df: pd.DataFrame) -> set:
    """
    从 limit_list_d.csv 读取涨停代码集合（含无后缀版本）
    """
    if df is None or df.empty:
        return set()

    df = _normalize_id_columns(df)
    ts_col = "ts_code" if "ts_code" in df.columns else None
    if ts_col is None:
        # 再兜底
        for c in ("TS_CODE", "code", "证券代码", "股票代码", "代码"):
            if c in df.columns:
                ts_col = c
                break
    if ts_col is None or ts_col not in df.columns:
        return set()

    out: set = set()
    for v in df[ts_col].astype(str).tolist():
        v = _safe_str(v)
        if not v:
            continue
        out.add(v)
        out.add(_to_nosuffix(v))
    return out


def _read_limit_list_anyway(s: Settings, trade_date: str, warnings: List[str]) -> pd.DataFrame:
    """
    读取指定 trade_date 的 limit_list_d 数据（强韧）：
    1) s.data_repo.read_limit_list(trade_date)（若存在）
    2) _warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD/limit_list_d.csv
    3) _warehouse/a-share-top3-data/data/raw/YYYY/limit_list_d.csv（聚合文件，若存在则过滤）
    """
    # 1) data_repo
    dr = getattr(s, "data_repo", None)
    if dr is not None and hasattr(dr, "read_limit_list"):
        try:
            df = dr.read_limit_list(trade_date)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            warnings.append(f"read_limit_list via s.data_repo failed: {type(e).__name__}")

    # 2) per-day snapshot file
    try:
        y = str(trade_date)[:4]
        p = Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / str(trade_date) / "limit_list_d.csv"
        if p.exists():
            return _read_csv_guess(p)
    except Exception:
        pass

    # 3) aggregated (rare)
    try:
        y = str(trade_date)[:4]
        p = Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / "limit_list_d.csv"
        if not p.exists():
            warnings.append(f"limit_list_d not found for {trade_date} (warehouse)")
            return pd.DataFrame()

        df = _read_csv_guess(p)
        if df.empty:
            return df

        # filter by trade_date column if exists
        for c in ("trade_date", "TRADE_DATE", "日期", "交易日期"):
            if c in df.columns:
                td = df[c].astype(str).str.replace("-", "").str.strip()
                return df[td == str(trade_date)].copy()
        return df
    except Exception as e:
        warnings.append(f"fallback read limit_list failed: {type(e).__name__}")
        return pd.DataFrame()


# ============================================================
# Feature history -> train set
# ============================================================

def _normalize_feature_columns(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    """
    把 feature_history 中可能的别名列映射为 FEATURES 标准列。
    不覆盖已有标准列。
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    lower_map = {str(c).strip().lower(): c for c in out.columns}

    for canon, aliases in FEATURE_ALIASES.items():
        if canon in out.columns:
            continue
        found = None
        if canon.lower() in lower_map:
            found = lower_map[canon.lower()]
        else:
            for a in aliases:
                ak = str(a).strip().lower()
                if ak in lower_map:
                    found = lower_map[ak]
                    break
        if found:
            out[canon] = out[found]
        else:
            out[canon] = ""

    # id/date
    if "trade_date" not in out.columns:
        for c in ("TRADE_DATE", "日期", "交易日期", "dt", "date"):
            if c in out.columns:
                out.rename(columns={c: "trade_date"}, inplace=True)
                break

    out = _normalize_id_columns(out)
    if "ts_code" not in out.columns:
        # 兜底
        for c in ("TS_CODE", "code", "证券代码", "股票代码", "代码"):
            if c in out.columns:
                out.rename(columns={c: "ts_code"}, inplace=True)
                break

    # clean
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "").str.strip()
    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].astype(str).str.strip()

    # warn if still missing essentials
    if "trade_date" not in out.columns:
        warnings.append("feature_history missing trade_date column.")
    if "ts_code" not in out.columns:
        warnings.append("feature_history missing ts_code column.")

    return out


def _load_feature_history(warnings: List[str]) -> pd.DataFrame:
    df = _read_csv_guess(FEATURE_HISTORY_PATH)
    if df is None or df.empty:
        warnings.append("feature_history.csv not found or empty: outputs/learning/feature_history.csv")
        return pd.DataFrame()
    return df


def _build_train_set_from_feature_history(
    s: Settings,
    lookback_days: int,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    从 feature_history.csv 构建训练集：
    - X: FEATURES
    - y: next_trade_date 是否涨停
    """
    meta: Dict[str, Any] = {
        "feature_history_file": str(FEATURE_HISTORY_PATH) if FEATURE_HISTORY_PATH.exists() else "",
        "lookback_days": int(lookback_days),
        "dates_total": 0,
        "dates_used": 0,
        "rows_raw": 0,
        "rows_used": 0,
        "rows_dropped_allzero": 0,
    }

    df0 = _load_feature_history(warnings)
    if df0.empty:
        return pd.DataFrame(), meta

    meta["rows_raw"] = int(len(df0))

    df = _normalize_feature_columns(df0, warnings)
    if df.empty or "trade_date" not in df.columns or "ts_code" not in df.columns:
        warnings.append("feature_history invalid after normalize (missing trade_date/ts_code).")
        return pd.DataFrame(), meta

    # keep valid date
    df = df[df["trade_date"].astype(str).str.match(r"^\d{8}$", na=False)].copy()
    df = df[df["ts_code"].astype(str).map(lambda x: len(_safe_str(x)) > 0)].copy()
    if df.empty:
        warnings.append("feature_history filtered empty (no valid trade_date/ts_code).")
        return pd.DataFrame(), meta

    # unique trading dates from feature_history
    uniq_dates = sorted(df["trade_date"].dropna().unique().tolist())
    uniq_dates = [d for d in uniq_dates if re.match(r"^\d{8}$", str(d))]
    meta["dates_total"] = int(len(uniq_dates))
    if not uniq_dates:
        warnings.append("feature_history has no valid YYYYMMDD trade_date.")
        return pd.DataFrame(), meta

    use_dates = uniq_dates[-int(lookback_days):] if len(uniq_dates) > lookback_days else uniq_dates
    meta["dates_used"] = int(len(use_dates))

    df = df[df["trade_date"].isin(use_dates)].copy()
    if df.empty:
        warnings.append("feature_history lookback filtered empty.")
        return pd.DataFrame(), meta

    # numeric features
    for c in FEATURES:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

    # drop all-zero feature rows (no information)
    xsum = df[FEATURES].abs().sum(axis=1)
    drop_mask = (xsum <= 1e-12)
    dropped = int(drop_mask.sum())
    if dropped > 0:
        meta["rows_dropped_allzero"] = dropped
        df = df[~drop_mask].copy()

    meta["rows_used"] = int(len(df))
    if df.empty:
        warnings.append("after dropping all-zero feature rows, train_df is empty.")
        return pd.DataFrame(), meta

    # build labels based on next_trade_date (from date sequence in feature_history)
    date_to_next: Dict[str, Optional[str]] = {}
    for d in use_dates:
        date_to_next[d] = _infer_next_trade_date(uniq_dates, d)

    # cache limit sets
    cache_limit: Dict[str, set] = {}

    labels: List[int] = []
    next_dates: List[str] = []

    for _, row in df.iterrows():
        d = str(row["trade_date"]).strip()
        code = str(row["ts_code"]).strip()
        nd = date_to_next.get(d) or ""
        next_dates.append(nd)

        if not nd:
            labels.append(0)
            continue

        if nd not in cache_limit:
            lim_df = _read_limit_list_anyway(s, nd, warnings)
            cache_limit[nd] = _limit_codes_from_df(lim_df)

        limset = cache_limit[nd]
        hit = (code in limset) or (_to_nosuffix(code) in limset)
        labels.append(1 if hit else 0)

    df["next_trade_date"] = next_dates
    df["label"] = labels

    # clean for training
    keep_cols = ["trade_date", "next_trade_date", "ts_code", "label"] + FEATURES
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    return df, meta


# ============================================================
# Train & Save models
# ============================================================

def _train_lr(train_df: pd.DataFrame, warnings: List[str]) -> Optional[Any]:
    if Pipeline is None or StandardScaler is None or LogisticRegression is None:
        warnings.append("sklearn not available: LR training skipped.")
        return None

    X = train_df[FEATURES].astype(float).values
    y = train_df["label"].astype(int).values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    model.fit(X, y)
    return model


def _train_lgbm(train_df: pd.DataFrame, warnings: List[str]) -> Optional[Any]:
    if lgb is None:
        warnings.append("lightgbm not available: LGBM training skipped.")
        return None

    X = train_df[FEATURES].astype(float).values
    y = train_df["label"].astype(int).values

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _save_joblib_model(model: Any, path: Path, warnings: List[str]) -> bool:
    if model is None:
        return False
    if joblib is None:
        warnings.append("joblib not available: cannot save model.")
        return False
    try:
        _ensure_dir(path.parent)
        joblib.dump(model, path)
        return True
    except Exception as e:
        warnings.append(f"save model failed: {path.name} ({type(e).__name__})")
        return False


def _train_and_save_models(
    s: Settings,
    train_df: pd.DataFrame,
    warnings: List[str],
) -> Dict[str, Any]:
    """
    训练并落盘 models/step5_lr.joblib / models/step5_lgbm.joblib
    """
    res: Dict[str, Any] = {
        "trained": False,
        "lr_saved": False,
        "lgbm_saved": False,
        "models_dir": "",
        "detail": {},
    }

    if train_df is None or train_df.empty:
        warnings.append("train_df empty: training skipped.")
        res["detail"] = {"reason": "train_df empty"}
        return res

    # basic class check
    pos = int(train_df["label"].astype(int).sum())
    neg = int(len(train_df) - pos)

    res["detail"]["train_rows"] = int(len(train_df))
    res["detail"]["pos"] = pos
    res["detail"]["neg"] = neg

    if len(train_df) < MIN_SAMPLES:
        warnings.append(f"not enough samples for cold start: n={len(train_df)} < {MIN_SAMPLES}")
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

    # Train LR (must)
    lr_model = _train_lr(train_df, warnings)
    lr_path = models_dir / LR_MODEL_PATH.name
    lr_saved = _save_joblib_model(lr_model, lr_path, warnings)
    res["lr_saved"] = bool(lr_saved)

    # Train LGBM (optional)
    lgbm_model = _train_lgbm(train_df, warnings)
    lgbm_path = models_dir / LGBM_MODEL_PATH.name
    lgbm_saved = _save_joblib_model(lgbm_model, lgbm_path, warnings)
    res["lgbm_saved"] = bool(lgbm_saved)

    res["trained"] = bool(lr_saved or lgbm_saved)
    res["detail"]["lr_path"] = str(lr_path) if lr_saved else ""
    res["detail"]["lgbm_path"] = str(lgbm_path) if lgbm_saved else ""
    return res


# ============================================================
# Main Step7
# ============================================================

def run_step7(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    主入口：被 pipeline 调用
    """
    warnings: List[str] = []

    outputs_dir = _get_outputs_dir(s)
    learning_dir = outputs_dir / "learning"
    _ensure_dir(outputs_dir)
    _ensure_dir(learning_dir)

    # topn / lookback
    topn = 10
    try:
        topn = int(getattr(s, "topn", 10) or 10)
    except Exception:
        topn = 10

    lookback_days = 150
    try:
        lookback_days = int(getattr(s, "step7_lookback_days", getattr(s, "lookback_days", 150)) or 150)
    except Exception:
        lookback_days = 150

    # ---------------------------
    # 1) 命中率统计（基于历史 predict_top10）
    # ---------------------------
    predict_files = _list_predict_files(outputs_dir)
    predict_dates = [d for d in [_trade_date_from_output_path(p) for p in predict_files] if d]
    predict_dates = sorted(list(dict.fromkeys(predict_dates)))  # uniq

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

        nd = _infer_next_trade_date(predict_dates, d)
        codes = _parse_topn_codes(md_path, topn=topn)

        if not nd:
            hit_rows.append({
                "trade_date": d,
                "next_trade_date": "",
                "topn": len(codes),
                "hit": "",
                "hit_rate": "",
                "note": "no next_trade_date",
            })
            continue

        lim_df = _read_limit_list_anyway(s, nd, warnings)
        lim_set = _limit_codes_from_df(lim_df)

        hit = 0
        for c in codes:
            if (c in lim_set) or (_to_nosuffix(c) in lim_set):
                hit += 1

        hit_rate = hit / max(1, len(codes))
        hit_rows.append({
            "trade_date": d,
            "next_trade_date": nd,
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
    # 2) 构建训练集（来自 feature_history.csv）并训练落盘模型
    # ---------------------------
    train_df, train_meta = _build_train_set_from_feature_history(
        s=s,
        lookback_days=lookback_days,
        warnings=warnings,
    )

    # 训练执行（保证落盘 models/*.joblib）
    train_result = _train_and_save_models(s=s, train_df=train_df, warnings=warnings)

    # ---------------------------
    # 3) 输出学习报告
    # ---------------------------
    latest_hit = None
    if not hit_df.empty:
        v = hit_df[hit_df["hit"].astype(str).str.len() > 0]
        if not v.empty:
            latest_hit = v.iloc[-1].to_dict()

    report = {
        "ts": _now_str(),
        "topn": topn,
        "lookback_days": lookback_days,
        "latest_hit": latest_hit,
        "train_meta": train_meta,
        "train_result": train_result,
        "warnings": warnings,
    }

    report_json = learning_dir / "step7_report_latest.json"
    _safe_write_json(report_json, report)

    md_lines: List[str] = []
    md_lines.append("# Step7 自学习报告（latest）")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{report['ts']}")
    md_lines.append(f"- TopN：{topn}")
    md_lines.append(f"- Lookback：{lookback_days} 天")
    md_lines.append("")

    md_lines.append("## 1) 最新命中")
    if latest_hit:
        md_lines.append("")
        md_lines.append(f"- trade_date：{latest_hit.get('trade_date','')}")
        md_lines.append(f"- next_trade_date：{latest_hit.get('next_trade_date','')}")
        md_lines.append(f"- hit/topn：{latest_hit.get('hit','')}/{latest_hit.get('topn','')}")
        md_lines.append(f"- hit_rate：{latest_hit.get('hit_rate','')}")
    else:
        md_lines.append("")
        md_lines.append("- 暂无可验证命中（缺少下一交易日预测文件或数据）")

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


# CLI 调试入口（不会影响主 pipeline）
if __name__ == "__main__":
    s = Settings()
    out = run_step7(s, ctx={})
    print(json.dumps(out, ensure_ascii=False, indent=2))
