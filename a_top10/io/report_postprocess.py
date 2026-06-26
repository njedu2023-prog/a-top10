from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.io.writers import (
    _canonicalize_prediction_frame,
    _df_to_md_table,
    _format_probability,
    _format_score,
)


PERFORMANCE_START_VERIFY_DATE = "20260626"

CANDIDATE_REPORT_COLS = [
    "rank",
    "ts_code",
    "name",
    "final_score_v2",
    "final_score_base",
    "Probability",
    "strength_plus_score",
    "board",
    "ThemeBoost",
    "intraday_quality_score",
    "intraday_soft_risk_score",
    "intraday_hard_risk_flag",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "auction_strength_score",
    "intraday_coverage",
    "risk_level",
    "risk_label",
]

PROFESSIONAL_BUCKETS: List[Tuple[str, int]] = [
    ("Top1", 1),
    ("Top1-3", 3),
    ("Top1-5", 5),
    ("Top10", 10),
]

RANK_EFFECTIVENESS_WEIGHTS = {
    "Top1": 0.40,
    "Top1-3": 0.30,
    "Top1-5": 0.20,
    "Top10": 0.10,
}


def _clean_date_value(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return text
    if re.fullmatch(r"\d{8}\.0", text):
        return text.split(".")[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _read_csv_guess(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def _fmt_ret(value: Optional[float]) -> str:
    return _fmt_pct(value)


def _load_payload(outdir: Path, trade_date: str) -> Dict[str, Any]:
    path = outdir / f"predict_top10_{trade_date}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_candidate_pool(payload: Dict[str, Any]) -> pd.DataFrame:
    full = _canonicalize_prediction_frame(pd.DataFrame(payload.get("full") or []))
    topn = _canonicalize_prediction_frame(pd.DataFrame(payload.get("topN") or []))
    if full.empty:
        return pd.DataFrame()

    top_codes = set(topn.get("ts_code", pd.Series(dtype=str)).astype(str).tolist())
    out = full[~full["ts_code"].astype(str).isin(top_codes)].copy()
    if out.empty:
        return out

    out = out.sort_values(
        by=["Probability", "StrengthScore"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["rank"] = range(11, len(out) + 11)
    if "intraday_available" in out.columns:
        out["intraday_coverage"] = (
            pd.to_numeric(out["intraday_available"], errors="coerce")
            .fillna(0)
            .map(lambda x: "是" if x > 0 else "否")
        )
    return out


def _format_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "Probability",
        "StrengthScore",
        "ThemeBoost",
        "final_score_v2",
        "final_score_base",
        "strength_plus_score",
        "intraday_quality_score",
        "intraday_soft_risk_score",
        "intraday_risk_score",
        "late_withdraw_score",
        "reseal_score",
        "auction_strength_score",
    ]:
        if col in out.columns:
            out[col] = out[col].map(_format_probability if col == "Probability" else _format_score)
    return out


def _replace_section(md_text: str, heading: str, body: str) -> str:
    marker = f"## {heading}"
    start = md_text.find(marker)
    if start < 0:
        return md_text

    next_start = md_text.find("\n## ", start + len(marker))
    replacement = f"{marker}\n{body.rstrip()}\n\n"
    if next_start < 0:
        return md_text[:start] + replacement
    return md_text[:start] + replacement + md_text[next_start + 1 :]


def _replace_section_with_heading(md_text: str, old_heading: str, new_heading: str, body: str) -> str:
    marker = f"## {old_heading}"
    start = md_text.find(marker)
    replacement = f"## {new_heading}\n{body.rstrip()}\n\n"
    if start < 0:
        return md_text.rstrip() + "\n\n" + replacement

    next_start = md_text.find("\n## ", start + len(marker))
    if next_start < 0:
        return md_text[:start] + replacement
    return md_text[:start] + replacement + md_text[next_start + 1 :]


def _update_markdown(path: Path, heading: str, body: str) -> bool:
    if not path.exists():
        return False
    old = path.read_text(encoding="utf-8")
    new = _replace_section(old, heading, body)
    if new == old:
        return False
    path.write_text(new, encoding="utf-8")
    print(f"[WRITE] {path} (candidate table postprocess)")
    return True


def _update_markdown_section(path: Path, old_heading: str, new_heading: str, body: str, tag: str) -> bool:
    if not path.exists():
        return False
    old = path.read_text(encoding="utf-8")
    new = _replace_section_with_heading(old, old_heading, new_heading, body)
    if new == old:
        return False
    path.write_text(new, encoding="utf-8")
    print(f"[WRITE] {path} ({tag})")
    return True


def ensure_candidate_pool_report_columns(outdir: Path, trade_date: str) -> None:
    payload = _load_payload(outdir, trade_date)
    if not payload:
        print(f"[WARN] candidate table postprocess skipped: missing payload for {trade_date}")
        return

    verify_date = str(payload.get("verify_date") or "").strip()
    if not verify_date:
        print(f"[WARN] candidate table postprocess skipped: missing verify_date for {trade_date}")
        return

    candidate_df = _format_candidate_pool(_build_candidate_pool(payload))
    if candidate_df.empty:
        body = "（无 Top10 之外的候选样本）"
    else:
        body = _df_to_md_table(candidate_df, cols=CANDIDATE_REPORT_COLS)

    heading = f"{trade_date} 预测：{verify_date} 候选池补充表"
    updated: List[bool] = [
        _update_markdown(outdir / f"predict_top10_{trade_date}.md", heading, body),
        _update_markdown(outdir / "latest.md", heading, body),
    ]
    if not any(updated):
        print(f"[WARN] candidate table postprocess found no section for {trade_date}")


def _load_feature_history(outdir: Path) -> pd.DataFrame:
    return _read_csv_guess(outdir / "learning" / "feature_history.csv")


def _rank_day(df_day: pd.DataFrame) -> pd.DataFrame:
    day = df_day.copy()
    if "Probability" in day.columns:
        day["_rank_prob"] = pd.to_numeric(day["Probability"], errors="coerce")
    else:
        day["_rank_prob"] = pd.NA
    if "rank" in day.columns:
        day["_rank_order"] = pd.to_numeric(day["rank"], errors="coerce")
    else:
        day["_rank_order"] = pd.NA
    day = day.sort_values(
        by=["_rank_prob", "_rank_order"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return day.drop(columns=["_rank_prob", "_rank_order"], errors="ignore")


def _build_professional_stats(feature_history: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "start_verify_date": PERFORMANCE_START_VERIFY_DATE,
        "eligible_days": 0,
        "eligible_rows": 0,
        "rank_effectiveness_weights": RANK_EFFECTIVENESS_WEIGHTS,
    }
    if feature_history is None or feature_history.empty:
        return pd.DataFrame(), None, meta

    df = feature_history.copy()
    for col in ["trade_date", "verify_date", "Probability", "y_limit_hit", "y_next_ret"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["trade_date"] = df["trade_date"].map(_clean_date_value)
    df["verify_date"] = df["verify_date"].map(_clean_date_value)
    df["y_limit_hit"] = pd.to_numeric(df["y_limit_hit"], errors="coerce")
    df["y_next_ret"] = pd.to_numeric(df["y_next_ret"], errors="coerce")
    df = df[
        (df["verify_date"] >= PERFORMANCE_START_VERIFY_DATE)
        & df["trade_date"].ne("")
        & df["y_limit_hit"].notna()
    ].copy()

    if df.empty:
        return pd.DataFrame(), None, meta

    parts: Dict[str, List[pd.DataFrame]] = {label: [] for label, _ in PROFESSIONAL_BUCKETS}
    for _, day in df.groupby("trade_date", sort=True):
        ranked = _rank_day(day)
        for label, n in PROFESSIONAL_BUCKETS:
            parts[label].append(ranked.head(n).copy())

    rows: List[Dict[str, Any]] = []
    hit_rates: Dict[str, float] = {}
    for label, _ in PROFESSIONAL_BUCKETS:
        selected = pd.concat(parts[label], ignore_index=True) if parts[label] else pd.DataFrame()
        if selected.empty:
            rows.append({
                "档位": label,
                "样本日数": 0,
                "预测样本数": 0,
                "涨停命中数": 0,
                "涨停命中率": "",
                "上涨胜率": "",
                "平均涨幅": "",
            })
            continue

        y = pd.to_numeric(selected["y_limit_hit"], errors="coerce")
        ret = pd.to_numeric(selected["y_next_ret"], errors="coerce")
        sample_count = int(y.notna().sum())
        hit_count = int(y.fillna(0).sum())
        hit_rate = float(hit_count / sample_count) if sample_count else 0.0
        ret_valid = ret.dropna()
        up_rate = float((ret_valid > 0).mean()) if len(ret_valid) else None
        avg_ret = float(ret_valid.mean()) if len(ret_valid) else None
        hit_rates[label] = hit_rate
        rows.append({
            "档位": label,
            "样本日数": int(selected["trade_date"].nunique()),
            "预测样本数": sample_count,
            "涨停命中数": hit_count,
            "涨停命中率": _fmt_pct(hit_rate),
            "上涨胜率": _fmt_pct(up_rate),
            "平均涨幅": _fmt_ret(avg_ret),
        })

    used_weights = {k: v for k, v in RANK_EFFECTIVENESS_WEIGHTS.items() if k in hit_rates}
    weight_sum = sum(used_weights.values())
    effectiveness = None
    if weight_sum > 0:
        effectiveness = sum(hit_rates[k] * w for k, w in used_weights.items()) / weight_sum

    meta["eligible_days"] = int(df["trade_date"].nunique())
    meta["eligible_rows"] = int(len(df))
    return pd.DataFrame(rows), effectiveness, meta


def _professional_stats_body(stats_df: pd.DataFrame, effectiveness: Optional[float], meta: Dict[str, Any]) -> str:
    lines = [
        f"统计口径：只统计 `verify_date >= {PERFORMANCE_START_VERIFY_DATE}` 的已验证样本；按预测日内 `Probability` 降序重新取 Top1 / Top1-3 / Top1-5 / Top10。",
        "",
    ]
    if stats_df.empty:
        lines.append("暂无满足新口径的已验证样本。")
        return "\n".join(lines)

    lines.append(_df_to_md_table(stats_df))
    lines.append("")
    lines.append("排序有效性口径：`Top1 * 40% + Top1-3 * 30% + Top1-5 * 20% + Top10 * 10%`。")
    lines.append(f"排序有效性：**{_fmt_pct(effectiveness)}**")
    lines.append("")
    lines.append(f"样本范围：{meta.get('eligible_days', 0)} 个验证日，{meta.get('eligible_rows', 0)} 条候选样本。")
    return "\n".join(lines)


def ensure_professional_performance_sections(outdir: Path, trade_date: str) -> None:
    stats_df, effectiveness, meta = _build_professional_stats(_load_feature_history(outdir))
    body = _professional_stats_body(stats_df, effectiveness, meta)

    old_daily_heading = "近10日 Top10 绩效"
    new_daily_heading = f"专业分层回测统计（{PERFORMANCE_START_VERIFY_DATE} 起）"
    daily_paths = [
        outdir / f"predict_top10_{trade_date}.md",
        outdir / "latest.md",
    ]
    updated = [
        _update_markdown_section(p, old_daily_heading, new_daily_heading, body, "professional performance postprocess")
        for p in daily_paths
    ]

    learning_path = outdir / "learning" / "step7_report_latest.md"
    learning_updated = _update_markdown_section(
        learning_path,
        "1.1) 近10日 Top10 命中率（done-only）",
        f"1.1) 专业分层回测统计（{PERFORMANCE_START_VERIFY_DATE} 起）",
        body,
        "professional learning report postprocess",
    )

    if not any(updated) and not learning_updated:
        print("[WARN] professional performance postprocess found no report sections")
