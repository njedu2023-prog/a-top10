#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def infer_trade_date(df: pd.DataFrame) -> str:
    env = str(os.environ.get("TRADE_DATE", "")).strip()
    if env:
        return env
    if "trade_date" in df.columns and df["trade_date"].notna().any():
        return str(df["trade_date"].dropna().astype(str).iloc[0])
    return ""


def stage_path(trade_date: str) -> Path:
    y = trade_date[:4]
    return Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / trade_date / "limit_stage.csv"


def merge_stage(df: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame:
    if df.empty or stage.empty or "ts_code" not in df.columns:
        return df
    s = stage.copy()
    keep = [c for c in ["ts_code", "晋阶", "limit_times", "advance_stage", "stage_quality_weight", "stage_risk_weight", "stage_prior", "stage_source", "up_stat"] if c in s.columns]
    s = s[keep].drop_duplicates("ts_code", keep="last")
    out = df.drop(columns=[c for c in keep if c != "ts_code" and c in df.columns], errors="ignore")
    out = out.merge(s, on="ts_code", how="left")
    if "晋阶" not in out.columns:
        out["晋阶"] = ""
    if "advance_stage" in out.columns:
        stage_txt = out["晋阶"].astype(str).str.strip()
        out["晋阶"] = stage_txt.where(stage_txt.ne("") & stage_txt.ne("nan") & stage_txt.ne("<NA>"), out["advance_stage"])
    return out


def _numeric_column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce").astype("float64")


def _canonical_probability(df: pd.DataFrame) -> pd.Series:
    for name in ("prob_final", "Probability", "prob", "probability"):
        if name in df.columns:
            probability = pd.to_numeric(df[name], errors="coerce")
            if probability.notna().any():
                return probability.fillna(0.0).clip(0.0, 1.0).astype("float64")
    raise ValueError("stage overlay requires prob_final or a compatible probability field")


def _resolve_pre_stage_score(df: pd.DataFrame) -> pd.Series:
    pre_stage = pd.Series(float("nan"), index=df.index, dtype="float64")

    if "final_score_pre_stage" in df.columns:
        pre_stage = _numeric_column(df, "final_score_pre_stage", float("nan"))

    if "raw_final_score" in df.columns:
        reconstructed = (
            _numeric_column(df, "raw_final_score")
            + _numeric_column(df, "intraday_bonus")
            - _numeric_column(df, "intraday_soft_risk_penalty")
            - _numeric_column(df, "intraday_hard_risk_penalty")
        )
        pre_stage = pre_stage.fillna(reconstructed)

    score_col = "final_score_v2" if "final_score_v2" in df.columns else "final_score"
    current_score = _numeric_column(df, score_col)
    if "stage_adjustment" in df.columns:
        current_score = current_score - _numeric_column(df, "stage_adjustment")
    return pre_stage.fillna(current_score).fillna(0.0).astype("float64")


def apply_stage_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_order = pd.Series(range(len(out)), index=out.index, dtype="int64")
    pre_stage = _resolve_pre_stage_score(out)
    q = _numeric_column(out, "stage_quality_weight", 1.0).fillna(1.0)
    r = _numeric_column(out, "stage_risk_weight", 0.0).fillna(0.0)
    bonus = ((q - 1.0) * 0.10).clip(-0.030, 0.015)
    penalty = (r * 0.055).clip(0.0, 0.018)
    out["stage_bonus"] = bonus.round(6)
    out["stage_risk_penalty"] = penalty.round(6)
    out["stage_adjustment"] = (bonus - penalty).round(6)
    out["final_score_pre_stage"] = pre_stage
    out["final_score_v2"] = (pre_stage + out["stage_adjustment"]).clip(0.0, 1.0).round(6)
    out["stage_adjustment_applied"] = True
    out["final_score"] = out["final_score_v2"]
    if "raw_final_score" not in out.columns:
        out["raw_final_score"] = pre_stage

    probability = _canonical_probability(out)
    out["prob_final"] = probability
    out["Probability"] = probability
    out["prob"] = probability

    out["_rank_probability"] = probability.fillna(float("-inf"))
    out["_rank_ts_code"] = out.get("ts_code", pd.Series("", index=out.index)).fillna("").astype(str)
    out["_rank_name"] = out.get("name", pd.Series("", index=out.index)).fillna("").astype(str)
    out["_rank_board"] = out.get("board", pd.Series("", index=out.index)).fillna("").astype(str)
    out["_source_order"] = source_order
    out = out.sort_values(
        ["_rank_probability", "_rank_ts_code", "_rank_name", "_rank_board", "_source_order"],
        ascending=[False, True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    if not out["prob_final"].is_monotonic_decreasing:
        raise AssertionError("ranking contract violated: prob_final must be descending")
    out["rank"] = range(1, len(out) + 1)
    out["rank_v2"] = out["rank"]
    out = out.drop(columns=["_rank_probability", "_rank_ts_code", "_rank_name", "_rank_board", "_source_order"])
    front = ["trade_date", "verify_date", "rank", "ts_code", "name", "晋阶"]
    cols = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    return out[cols]


def write_outputs(df: pd.DataFrame, trade_date: str) -> None:
    Path("outputs/decisio").mkdir(parents=True, exist_ok=True)
    Path("outputs/learning").mkdir(parents=True, exist_ok=True)
    full = df.copy()
    top10 = df.head(10).copy()
    for path, data in [
        (Path("outputs/decisio/pred_source_latest.csv"), full),
        (Path(f"outputs/decisio/pred_source_{trade_date}.csv"), full),
        (Path("outputs/decisio/pred_decisio_latest.csv"), full),
        (Path(f"outputs/decisio/pred_decisio_{trade_date}.csv"), full),
        (Path("outputs/learning/pred_top10_latest.csv"), top10),
        (Path(f"outputs/learning/pred_top10_{trade_date}.csv"), top10),
    ]:
        data.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[stage_overlay] wrote {path} rows={len(data)}")


def to_joinquant_code(value: object) -> str:
    code = str(value).strip().upper()
    if code.endswith(".SH"):
        return code[:-3] + ".XSHG"
    if code.endswith(".SZ"):
        return code[:-3] + ".XSHE"
    if code.endswith(".BJ"):
        return code[:-3] + ".XBEI"
    code6 = code.split(".")[0]
    if code6.startswith(("600", "601", "603", "605", "688", "689")):
        return code6 + ".XSHG"
    if code6.startswith(("000", "001", "002", "003", "300", "301")):
        return code6 + ".XSHE"
    return code


def write_top3(df: pd.DataFrame, trade_date: str) -> None:
    top3 = df.head(3).copy()
    weight = 1.0 / len(top3) if len(top3) else 0.0
    regime = top3.get("regime_state", pd.Series("neutral", index=top3.index)).astype(str).str.strip().str.upper()
    regime = regime.where(regime.isin(["RISK_ON", "RISK_OFF", "NEUTRAL"]), "NEUTRAL")
    jq = pd.DataFrame({
        "trade_date": top3["trade_date"] if "trade_date" in top3.columns else trade_date,
        "target_trade_date": top3["verify_date"] if "verify_date" in top3.columns else "",
        "jq_code": top3["ts_code"].map(to_joinquant_code) if "ts_code" in top3.columns else "",
        "target_weight": weight,
        "risk_budget": 1.0,
        "regime": regime,
        "reason": "a_top10_probability_rank_1_3",
    })
    Path("outputs").mkdir(parents=True, exist_ok=True)
    jq.to_csv("outputs/top101-3.csv", index=False, encoding="utf-8-sig")
    jq.to_csv(f"outputs/top101-3_{trade_date}.csv", index=False, encoding="utf-8-sig")
    print(f"[stage_overlay] wrote top101-3 rows={len(jq)}")


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    mapping = {
        "rank": "排名",
        "ts_code": "代码",
        "name": "股票",
        "final_score_v2": "最终分",
        "raw_final_score": "原始分",
        "prob": "涨停概率",
        "Probability": "涨停概率",
        "p_limit_up_calibrated": "续板概率",
        "rank_score": "排序分",
        "StrengthScore": "强度分",
        "board": "行业版块",
        "ThemeBoost": "题材加成",
        "intraday_quality_score": "分时质量",
        "intraday_soft_risk_score": "软风险",
        "intraday_hard_risk_flag": "硬风险",
        "late_withdraw_score": "尾盘风险",
        "reseal_score": "回封分",
        "open_board_count": "炸板数",
        "auction_strength_score": "竞价强度",
        "risk_level": "风险级别",
        "risk_label": "风险标签",
    }
    calibrated = bool(
        "probability_is_calibrated" in show.columns
        and pd.to_numeric(show["probability_is_calibrated"], errors="coerce").fillna(0).eq(1).all()
        and "p_limit_up_calibrated" in show.columns
        and pd.to_numeric(show["p_limit_up_calibrated"], errors="coerce").notna().all()
    )
    display_probability = "p_limit_up_calibrated" if calibrated else (
        "rank_score" if "rank_score" in show.columns else "Probability"
    )
    cols = [c for c in ["rank", "ts_code", "name", "晋阶", "final_score_v2", display_probability, "StrengthScore", "board", "ThemeBoost", "intraday_quality_score", "intraday_soft_risk_score", "intraday_hard_risk_flag", "late_withdraw_score", "reseal_score", "open_board_count", "auction_strength_score", "risk_level", "risk_label"] if c in show.columns]
    show = show[cols].rename(columns=mapping)
    return show.to_markdown(index=False)


def replace_section(md: str, keyword: str, table: str) -> str:
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and keyword in line:
            start = i
            break
    if start is None:
        return md
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    heading = lines[start]
    new_lines = lines[: start + 1] + ["", table, ""] + lines[end:]
    return "\n".join(new_lines) + "\n"


def update_markdown(df: pd.DataFrame, trade_date: str) -> None:
    top = markdown_table(df.head(10))
    tail = markdown_table(df.iloc[10:])
    for path in [Path("outputs/latest.md"), Path(f"outputs/predict_top10_{trade_date}.md")]:
        if not path.exists():
            continue
        md = path.read_text(encoding="utf-8")
        md = replace_section(md, "涨停 Top10", top)
        md = replace_section(md, "候选池补充表", tail)
        path.write_text(md, encoding="utf-8")
        print(f"[stage_overlay] updated markdown {path}")


def main() -> int:
    src = read_csv(Path("outputs/decisio/pred_source_latest.csv"))
    if src.empty:
        src = read_csv(Path("outputs/decisio/pred_decisio_latest.csv"))
    if src.empty:
        raise SystemExit("no pred source available")
    trade_date = infer_trade_date(src)
    sp = stage_path(trade_date)
    st = read_csv(sp)
    if st.empty:
        print(f"[stage_overlay] no limit_stage.csv at {sp}; skip")
        return 0
    out = apply_stage_score(merge_stage(src, st))
    write_outputs(out, trade_date)
    write_top3(out, trade_date)
    update_markdown(out, trade_date)
    print(f"[stage_overlay] trade_date={trade_date} stage_rows={len(st)} output_rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
