#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step7：自学习闭环更新模型权重（命中率滚动提升）

目标：
- 回收 outputs/predict_top10_*.md 的 TopN 预测
- 用 “下一交易日”的 limit_list_d.csv 打标（命中统计）
- 构建训练样本（优先使用历史 step4_theme.csv / step4_history.csv 之类的“特征历史落盘”）
- 调用 Step5 训练（LR + LightGBM）并落盘 models/
- 输出命中率历史与最新报告到 outputs/learning

注意：
- Step7 不能损坏主程序：任何缺文件/字段/签名不匹配都必须“降级不报错”，只写 warnings。
"""

from __future__ import annotations

import json
import re
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings
from a_top10.steps.step5_ml_probability import (
    _get_ts_code_col,
    _normalize_id_columns,
    train_step5_models,
)


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


def _get_attr(s: Any, names: List[str], default: Any = None) -> Any:
    for n in names:
        if hasattr(s, n):
            v = getattr(s, n)
            if v is not None:
                return v
    return default


def _as_path(x: Any, default: Path) -> Path:
    if x is None:
        return default
    try:
        return Path(x)
    except Exception:
        return default


def _to_nosuffix(ts: str) -> str:
    ts = str(ts).strip()
    if not ts:
        return ts
    return ts.split(".")[0]


def _trade_date_from_output_path(p: Path) -> Optional[str]:
    # 支持 predict_top10_YYYYMMDD.md / predict_top10_YYYYMMDD_*.md
    name = Path(p).name
    m = re.match(r"predict_top10_(\d{8})(?:_.*)?\.md$", name)
    return m.group(1) if m else None


def _parse_topn_codes(md_path: Path, topn: int) -> List[str]:
    """
    解析 md 表格中 TopN 的股票代码（尽量宽松）
    典型行：| 1 | 600000.SH | xxx | ...
    """
    try:
        text = Path(md_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    codes: List[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*\|\s*\d+\s*\|\s*([0-9A-Za-z\.\-_]+)\s*\|", line)
        if m:
            codes.append(m.group(1).strip())

    codes = [c for c in codes if c]
    return codes[: int(topn)]


def _limit_codes_from_df(df: pd.DataFrame) -> List[str]:
    """
    从 limit_list_d.csv 读取涨停代码列表（兼容字段名）
    """
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


def _read_limit_list_anyway(s: Settings, trade_date: str, warnings: List[str]) -> pd.DataFrame:
    """
    读取指定 trade_date 的 limit_list_d 数据：
    1) 优先走 s.data_repo.read_limit_list(trade_date)
    2) 兜底从 _warehouse/a-share-top3-data/data/raw/YYYY/limit_list_d.csv 读取并过滤
    """
    # 1) data_repo
    data_repo = getattr(s, "data_repo", None)
    if data_repo is not None and hasattr(data_repo, "read_limit_list"):
        try:
            df = data_repo.read_limit_list(trade_date)
            if isinstance(df, pd.DataFrame):
                return df
        except Exception as e:
            warnings.append(f"read_limit_list via s.data_repo failed: {e}")

    # 2) fallback to warehouse csv
    try:
        y = trade_date[:4]
        base = Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / "limit_list_d.csv"
        if not base.exists():
            warnings.append(f"limit_list_d.csv not found at {base}")
            return pd.DataFrame()

        df = pd.read_csv(base, dtype=str, encoding="utf-8", engine="python")
        # 过滤 trade_date 列（兼容字段名）
        cand_cols = ["trade_date", "TRADE_DATE", "日期", "交易日期"]
        use_col = None
        for c in cand_cols:
            if c in df.columns:
                use_col = c
                break
        if use_col is None:
            warnings.append("limit_list_d.csv has no trade_date column, use whole file as fallback.")
            return df

        df = df[df[use_col].astype(str).str.replace("-", "").str.strip() == str(trade_date)]
        return df

    except Exception as e:
        warnings.append(f"fallback read limit_list_d.csv failed: {e}")
        return pd.DataFrame()


def _list_predict_files(outputs_dir: Path) -> List[Path]:
    if not outputs_dir.exists():
        return []
    ps = sorted(outputs_dir.glob("predict_top10_*.md"))
    return [p for p in ps if p.is_file()]


def _infer_next_trade_date(predict_dates: List[str], cur_date: str) -> Optional[str]:
    """
    用已存在的预测日期序列推断“下一交易日”：
    - 预测文件是按交易日生成的，因此下一交易日通常是列表中的下一个日期
    """
    try:
        idx = predict_dates.index(cur_date)
    except ValueError:
        return None
    if idx + 1 >= len(predict_dates):
        return None
    return predict_dates[idx + 1]


# ============================================================
# Training dataset builder
# ============================================================

def _find_feature_history_file(outputs_dir: Path) -> Optional[Path]:
    """
    寻找“可训练的特征历史文件”：
    优先级：step4_theme.csv / step4_theme_history.csv / step4_history.csv / theme_history.csv
    """
    candidates = [
        outputs_dir / "step4_theme.csv",
        outputs_dir / "step4_theme_history.csv",
        outputs_dir / "step4_history.csv",
        outputs_dir / "theme_history.csv",
        outputs_dir / "step4_theme_all.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    # 再兜底：找最近的 step4_theme_*.csv
    ps = sorted(outputs_dir.glob("step4_theme_*.csv"))
    if ps:
        return ps[-1]
    return None


def _build_train_df_from_history(
    s: Settings,
    outputs_dir: Path,
    lookback_days: int,
    warnings: List[str],
) -> pd.DataFrame:
    """
    从特征历史落盘文件中构建训练集（带 label）：
    - 要求至少有：trade_date + ts_code(或 code)
    - label：下一交易日是否涨停（1/0）
    """
    hist_path = _find_feature_history_file(outputs_dir)
    if hist_path is None:
        warnings.append("no feature history file found in outputs/, training skipped.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(hist_path, dtype=str, encoding="utf-8", engine="python")
    except Exception as e:
        warnings.append(f"read feature history failed: {e}, training skipped.")
        return pd.DataFrame()

    if df is None or df.empty:
        warnings.append("feature history is empty, training skipped.")
        return pd.DataFrame()

    df = _normalize_id_columns(df)

    # trade_date
    date_col = None
    for c in ["trade_date", "TRADE_DATE", "日期", "交易日期"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        warnings.append("feature history has no trade_date column, training skipped.")
        return pd.DataFrame()

    # ts_code
    ts_col = _get_ts_code_col(df)
    if ts_col is None or ts_col not in df.columns:
        warnings.append("feature history has no ts_code column, training skipped.")
        return pd.DataFrame()

    # 标准化日期格式 YYYYMMDD
    df[date_col] = df[date_col].astype(str).str.replace("-", "").str.strip()
    df[ts_col] = df[ts_col].astype(str).str.strip()

    # 仅取最近 lookback_days 的交易日（按文件中的唯一日期序排序）
    uniq_dates = sorted([d for d in df[date_col].dropna().unique().tolist() if re.match(r"^\d{8}$", str(d))])
    if not uniq_dates:
        warnings.append("feature history has no valid YYYYMMDD dates, training skipped.")
        return pd.DataFrame()

    use_dates = uniq_dates[-int(lookback_days):] if len(uniq_dates) > lookback_days else uniq_dates
    df = df[df[date_col].isin(use_dates)].copy()
    if df.empty:
        warnings.append("feature history filtered to empty by lookback, training skipped.")
        return pd.DataFrame()

    # 构建 label：next_day limit_list 命中
    # 为每个 trade_date 缓存下一日涨停代码集合，避免重复读文件
    date_to_next: Dict[str, Optional[str]] = {}
    # 直接用 uniq_dates 推断 next_date：同一历史文件中日期序列的下一个
    for d in use_dates:
        date_to_next[d] = _infer_next_trade_date(uniq_dates, d)

    cache_limit_codes: Dict[str, set] = {}

    labels: List[int] = []
    for _, row in df.iterrows():
        d = str(row[date_col]).strip()
        code = str(row[ts_col]).strip()
        nd = date_to_next.get(d)
        if not nd:
            labels.append(0)
            continue

        if nd not in cache_limit_codes:
            lim_df = _read_limit_list_anyway(s, nd, warnings)
            lim_codes = set(_limit_codes_from_df(lim_df))
            cache_limit_codes[nd] = lim_codes

        hit = (code in cache_limit_codes[nd]) or (_to_nosuffix(code) in cache_limit_codes[nd])
        labels.append(1 if hit else 0)

    df["label"] = labels

    # 清理全空列
    for c in list(df.columns):
        if df[c].isna().all():
            df.drop(columns=[c], inplace=True)

    return df


def _call_train_step5_models_dynamic(
    s: Settings,
    train_df: pd.DataFrame,
    models_dir: Path,
    warnings: List[str],
) -> Dict[str, Any]:
    """
    动态适配 train_step5_models 的签名：
    常见可能：
      train_step5_models(s, train_df, models_dir=...)
      train_step5_models(s, train_df)
      train_step5_models(train_df, models_dir=...)
      train_step5_models(train_df)
    """
    result: Dict[str, Any] = {"trained": False, "detail": {}}

    if train_df is None or train_df.empty:
        warnings.append("train_df empty, skip train_step5_models.")
        return result

    try:
        sig = inspect.signature(train_step5_models)
        params = list(sig.parameters.keys())
    except Exception as e:
        warnings.append(f"inspect.signature(train_step5_models) failed: {e}")
        return result

    kwargs: Dict[str, Any] = {}
    args: List[Any] = []

    # 识别 models_dir 参数名
    for k in ["models_dir", "model_dir", "out_dir", "save_dir", "models_path", "model_path"]:
        if k in params:
            kwargs[k] = str(models_dir)

    # 识别 settings 参数
    if len(params) >= 1:
        p0 = params[0]
        # 常见：第一个就是 s/settings
        if p0 in ["s", "settings", "cfg", "config"]:
            args.append(s)
            # 第二个才是 train_df
            if len(params) >= 2:
                args.append(train_df)
        else:
            # 第一个就是 train_df
            args.append(train_df)
            # 第二个可能是 s
            if len(params) >= 2 and params[1] in ["s", "settings", "cfg", "config"]:
                args.append(s)

    # 如果还没塞进去 train_df，就尽量补
    if not any(isinstance(a, pd.DataFrame) for a in args):
        # 找 train_df 位置：看有没有名为 train_df/data/df
        for k in ["train_df", "df", "data"]:
            if k in params:
                kwargs[k] = train_df
                break
        else:
            args.append(train_df)

    try:
        out = train_step5_models(*args, **kwargs)
        result["trained"] = True
        result["detail"] = {"return": str(type(out))}
        return result
    except Exception as e:
        warnings.append(f"train_step5_models call failed: {e}")
        return result


# ============================================================
# Main Step7
# ============================================================

def run_step7(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    主入口：被 pipeline 调用
    """
    warnings: List[str] = []

    outputs_dir = _as_path(_get_attr(s, ["outputs_dir", "output_dir", "outputs_path"], None), Path("outputs"))
    learning_dir = outputs_dir / "learning"
    models_dir = _as_path(_get_attr(s, ["models_dir", "model_dir", "models_path"], None), outputs_dir / "models")

    _ensure_dir(outputs_dir)
    _ensure_dir(learning_dir)
    _ensure_dir(models_dir)

    topn = int(_get_attr(s, ["topn", "TOPN"], 10) or 10)
    lookback_days = int(_get_attr(s, ["step7_lookback_days", "lookback_days"], 150) or 150)

    # ---------------------------
    # 1) 命中率统计（基于历史 predict_top10）
    # ---------------------------
    predict_files = _list_predict_files(outputs_dir)
    predict_dates = [d for d in [_trade_date_from_output_path(p) for p in predict_files] if d]
    predict_dates = sorted(list(dict.fromkeys(predict_dates)))  # uniq keep order

    hit_rows: List[Dict[str, Any]] = []

    # 构建 date -> file（同一天可能多份，取最后一个）
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

        # 没有 next day，则无法验证命中
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
        lim_set = set(_limit_codes_from_df(lim_df))

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

    # 落盘命中率历史
    hit_csv = learning_dir / "step7_hit_rate_history.csv"
    if not hit_df.empty:
        hit_df.to_csv(hit_csv, index=False, encoding="utf-8-sig")

    # ---------------------------
    # 2) 构建训练集并训练（真闭环核心）
    # ---------------------------
    train_df = _build_train_df_from_history(
        s=s,
        outputs_dir=outputs_dir,
        lookback_days=lookback_days,
        warnings=warnings,
    )

    # 训练统计信息
    train_meta: Dict[str, Any] = {
        "train_rows": int(len(train_df)) if train_df is not None else 0,
        "pos": int(train_df["label"].astype(int).sum()) if (train_df is not None and "label" in train_df.columns and not train_df.empty) else 0,
        "neg": int((len(train_df) - train_df["label"].astype(int).sum())) if (train_df is not None and "label" in train_df.columns and not train_df.empty) else 0,
        "feature_history_file": str(_find_feature_history_file(outputs_dir)) if _find_feature_history_file(outputs_dir) else "",
    }

    train_result = _call_train_step5_models_dynamic(
        s=s,
        train_df=train_df,
        models_dir=models_dir,
        warnings=warnings,
    )

    # ---------------------------
    # 3) 输出学习报告
    # ---------------------------
    latest_hit = None
    if not hit_df.empty:
        # 取最近一个“可验证”的记录（hit 非空）
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

    # 同时输出一份 md，便于人看
    md_lines: List[str] = []
    md_lines.append(f"# Step7 自学习报告（latest）")
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
    md_lines.append(f"- 样本数：{train_meta.get('train_rows',0)}")
    md_lines.append(f"- 正样本：{train_meta.get('pos',0)}")
    md_lines.append(f"- 负样本：{train_meta.get('neg',0)}")
    md_lines.append(f"- 特征历史文件：{train_meta.get('feature_history_file','') or '未找到'}")

    md_lines.append("")
    md_lines.append("## 3) 训练执行结果")
    md_lines.append("")
    md_lines.append(f"- trained：{train_result.get('trained')}")
    md_lines.append(f"- detail：{train_result.get('detail')}")

    if warnings:
        md_lines.append("")
        md_lines.append("## 4) Warnings")
        md_lines.append("")
        for w in warnings[:50]:
            md_lines.append(f"- {w}")
        if len(warnings) > 50:
            md_lines.append(f"- ...（共 {len(warnings)} 条，仅展示前 50 条）")

    report_md = learning_dir / "step7_report_latest.md"
    _safe_write_text(report_md, "\n".join(md_lines) + "\n")

    return {
        "step7_learning": {
            "hit_history_csv": str(hit_csv),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "models_dir": str(models_dir),
            "trained": bool(train_result.get("trained")),
            "train_rows": train_meta.get("train_rows", 0),
            "warnings": warnings,
        }
    }


# CLI 调试入口（不会影响主 pipeline）
if __name__ == "__main__":
    s = Settings()  # 依赖你现有的 config.py / default.yml
    out = run_step7(s, ctx={})
    print(json.dumps(out, ensure_ascii=False, indent=2))
