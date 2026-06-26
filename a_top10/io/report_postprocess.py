from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from a_top10.io.writers import (
    _canonicalize_prediction_frame,
    _df_to_md_table,
    _format_probability,
    _format_score,
)


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
