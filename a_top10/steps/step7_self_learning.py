#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step7：自学习闭环更新模型权重（命中率滚动提升）
- 回收 outputs/predict_top10_*.md 的 TopN 预测
- 用 outputs 里的下一份 trade_date 对应 limit_list_d.csv 打标
- 调用 Step5 训练（LR + LightGBM）并落盘 models/
- 输出命中率历史与最新报告到 outputs/learning
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings
from a_top10.steps.step5_ml_probability import _get_ts_code_col, _normalize_id_columns, train_step5_models


def _to_nosuffix(ts: str) -> str:
    ts = str(ts).strip()
    if not ts:
        return ts
    return ts.split(".")[0]


def _trade_date_from_output_path(p: Path) -> Optional[str]:
    m = re.match(r"predict_top10_(\d{8})\.md$", Path(p).name)
    return m.group(1) if m else None


def _parse_topn_codes(md_path: Path, topn: int) -> List[str]:
    text = Path(md_path).read_text(encoding="utf-8")
    codes: List[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*\|\s*\d+\s*\|\s*([0-9A-Za-z\.\-]+)\s*\|", line)
        if m:
            codes.append(m.group(1).strip())
    codes = codes[: int(topn)]
    codes = [c for c in codes if c]
    return codes


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


def _load_limit_codes(limit_csv: Path) -> List[str]:
    """
    兼容旧逻辑：如果仍传入 CSV 路径，则从文件读取。
    新逻辑优先走 DataRepo.read_limit_list(next_d) 直接拿 DataFrame。
    """
    p = Path(limit_csv)
    if not p.exists():
        return []

    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(p, dtype=str, encoding="gbk")
        except Exception:
            return []

    return _limit_codes_from_df(df)


def _dedup_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["trade_date", "next_trade_date"]).drop_duplicates(
        subset=["trade_date", "next_trade_date"], keep="last"
    )
    return df


def run_step7_self_learning(s: Settings, lookback: int = 150, evaluate_days: int = 30) -> Dict:
    outputs_dir = Path(getattr(s.io, "outputs_dir", "outputs"))
    learning_dir = outputs_dir.joinpath("learning")
    learning_dir.mkdir(parents=True, exist_ok=True)

    model_info = train_step5_models(s, lookback=lookback, theme_file_name="step4_theme.csv")

    output_files = sorted(Path(outputs_dir).glob("predict_top10_*.md"))
    data: List[Tuple[str, Path]] = []
    for p in output_files:
        d = _trade_date_from_output_path(p)
        if d:
            data.append((d, p))

    data.sort(key=lambda x: x[0])

    metrics: Dict[str, Any] = {
        "ok": bool(model_info.get("ok")),
        "lookback": int(lookback),
        "outputs_files": int(len(data)),
        "evaluate_days": int(evaluate_days),
        "trained_models": model_info,
    }

    report_path = learning_dir.joinpath("step7_report_latest.json")

    if len(data) < 2:
        metrics["reason"] = "need at least 2 predict_top10 outputs"
        report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    topn = int(getattr(s.io, "topn", 10))
    records: List[Dict[str, Any]] = []

    for i in range(len(data) - 1):
        d, md_path = data[i]
        next_d, _ = data[i + 1]

        codes = _parse_topn_codes(md_path, topn=topn)
        if not codes:
            continue

        # ✅ 修复点：DataRepo 没有 path_limit_list()，应使用 read_limit_list()
        try:
            df_limit = s.data_repo.read_limit_list(next_d)  # type: ignore[attr-defined]
        except Exception:
            df_limit = pd.DataFrame()

        limit_codes = set(_limit_codes_from_df(df_limit))
        if not limit_codes:
            # 兜底：如果 read_limit_list 异常或返回空，允许未来用文件路径方案（不影响主流程）
            # 但这里不再调用不存在的 path_limit_list
            continue

        hits = [1 if (c in limit_codes or _to_nosuffix(c) in limit_codes) else 0 for c in codes]
        hit_rate = sum(hits) / float(len(hits) or 1)

        records.append(
            {
                "trade_date": d,
                "next_trade_date": next_d,
                "topn": topn,
                "hit_cnt": int(sum(hits)),
                "hit_rate": float(hit_rate),
            }
        )

    history_file = learning_dir.joinpath("step7_hit_rate_history.csv")
    existing = pd.DataFrame()
    if history_file.exists():
        try:
            existing = pd.read_csv(history_file)
        except Exception:
            existing = pd.DataFrame()

    new_df = pd.DataFrame(records)
    df_all = _dedup_history(pd.concat([existing, new_df], ignore_index=True, sort=False))

    if not df_all.empty:
        df_all = df_all.sort_values("trade_date")

    df_all.to_csv(history_file, index=False)

    if not df_all.empty:
        metrics["hit_rate_last_days"] = int(min(evaluate_days, len(df_all)))
        metrics["hit_rate_last"] = float(df_all["hit_rate"].tail(evaluate_days).mean())

    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    s = Settings()
    run_step7_self_learning(s)


if __name__ == "__main__":
    main()
