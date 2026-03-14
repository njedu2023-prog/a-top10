from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


V2_BASE_COLS = [
    "rank",
    "ts_code",
    "name",
    "prob_rule",
    "prob_ml",
    "prob_final",
    "prob",
    "Probability",
    "final_score",
    "score",
    "StrengthScore",
    "ThemeBoost",
    "board",
]

V2_META_COLS = [
    "trade_date",
    "verify_date",
    "run_id",
    "run_attempt",
    "commit_sha",
    "generated_at_utc",
]

V2_ALL_COLS = V2_META_COLS[:2] + V2_BASE_COLS + V2_META_COLS[2:]

V2_REQUIRED_HISTORY_COLS = {
    "trade_date",
    "verify_date",
    "rank",
    "ts_code",
    "name",
    "prob_final",
    "final_score",
}

CODE_COL_CANDIDATES = ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]
NAME_COL_CANDIDATES = ["name", "stock_name", "名称", "股票", "证券名称", "股票简称"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"]


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _read_csv_guess(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_map.get(str(name).lower())
        if hit:
            return hit
    return None


def _normalize_rank(df: pd.DataFrame) -> pd.Series:
    if "rank" in df.columns:
        s = pd.to_numeric(df["rank"], errors="coerce")
        if s.notna().any():
            return s
    if "排名" in df.columns:
        s = pd.to_numeric(df["排名"], errors="coerce")
        if s.notna().any():
            return s
    return pd.Series(range(1, len(df) + 1), index=df.index, dtype="int64")


def _canonicalize_v2_frame(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=V2_BASE_COLS)

    src = df.copy()
    out = pd.DataFrame(index=src.index)

    code_col = _first_existing_col(src, CODE_COL_CANDIDATES)
    name_col = _first_existing_col(src, NAME_COL_CANDIDATES)
    board_col = _first_existing_col(src, BOARD_COL_CANDIDATES)

    out["rank"] = _normalize_rank(src)
    out["ts_code"] = src[code_col] if code_col else ""
    out["name"] = src[name_col] if name_col else ""
    out["board"] = src[board_col] if board_col else ""

    out["prob_rule"] = src["prob_rule"] if "prob_rule" in src.columns else ""
    out["prob_ml"] = src["prob_ml"] if "prob_ml" in src.columns else ""
    out["prob_final"] = src["prob_final"] if "prob_final" in src.columns else ""
    out["final_score"] = src["final_score"] if "final_score" in src.columns else ""

    out["StrengthScore"] = src["StrengthScore"] if "StrengthScore" in src.columns else (
        src["强度得分"] if "强度得分" in src.columns else ""
    )
    out["ThemeBoost"] = src["ThemeBoost"] if "ThemeBoost" in src.columns else (
        src["题材加成"] if "题材加成" in src.columns else ""
    )

    for col in ["prob_rule", "prob_ml", "prob_final", "final_score", "StrengthScore", "ThemeBoost"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["prob"] = out["prob_final"]
    out["Probability"] = out["prob_final"]
    out["score"] = out["final_score"]
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")

    return out.reindex(columns=V2_BASE_COLS, fill_value="")


def _enrich_v2_with_meta(
    df: Optional[pd.DataFrame],
    trade_date: str,
    verify_date: str,
    run_meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    run_meta = run_meta or {}
    base = _canonicalize_v2_frame(df)
    out = base.copy()

    out.insert(0, "trade_date", trade_date)
    out.insert(1, "verify_date", verify_date)
    out["run_id"] = _safe_str(run_meta.get("run_id"))
    out["run_attempt"] = _safe_str(run_meta.get("run_attempt"))
    out["commit_sha"] = _safe_str(run_meta.get("commit_sha"))
    out["generated_at_utc"] = _safe_str(
        run_meta.get("generated_at_utc") or run_meta.get("run_time_utc")
    )

    return out.reindex(columns=V2_ALL_COLS, fill_value="")


def _is_v2_compatible_schema(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(c).strip() for c in df.columns}
    return V2_REQUIRED_HISTORY_COLS.issubset(cols)


def _load_json_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _migrate_from_json(json_path: Path) -> pd.DataFrame:
    payload = _load_json_payload(json_path)
    if not payload:
        return pd.DataFrame(columns=V2_ALL_COLS)

    trade_date = _safe_str(payload.get("trade_date"))
    verify_date = _safe_str(payload.get("verify_date"))
    topn = payload.get("topN") or []
    run_meta = payload.get("run_meta") or {}

    topn_df = pd.DataFrame(topn)
    return _enrich_v2_with_meta(
        topn_df,
        trade_date=trade_date,
        verify_date=verify_date,
        run_meta=run_meta,
    )


def _migrate_one_day(outputs_dir: Path, learning_dir: Path, trade_date: str) -> Optional[pd.DataFrame]:
    daily_csv = learning_dir / f"pred_top10_{trade_date}.csv"
    if daily_csv.exists():
        df = _read_csv_guess(daily_csv)
        if _is_v2_compatible_schema(df):
            fixed = df.copy()
            for c in V2_ALL_COLS:
                if c not in fixed.columns:
                    fixed[c] = ""
            fixed = fixed.reindex(columns=V2_ALL_COLS, fill_value="")
            fixed.to_csv(daily_csv, index=False, encoding="utf-8-sig")
            print(f"[KEEP] {daily_csv} rows={len(fixed)}")
            return fixed

    json_path = outputs_dir / f"predict_top10_{trade_date}.json"
    if not json_path.exists():
        print(f"[MISS] no usable source for {trade_date}")
        return None

    migrated = _migrate_from_json(json_path)
    if migrated.empty:
        print(f"[MISS] json empty for {trade_date}")
        return None

    migrated.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    print(f"[MIGRATE] {json_path.name} -> {daily_csv} rows={len(migrated)}")
    return migrated


def _rebuild_history_from_daily(learning_dir: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    skipped: List[str] = []

    for p in sorted(learning_dir.glob("pred_top10_*.csv")):
        m = re.match(r"pred_top10_(\d{8})\.csv$", p.name)
        if not m:
            continue

        df = _read_csv_guess(p)
        if df.empty:
            continue

        if not _is_v2_compatible_schema(df):
            skipped.append(p.name)
            continue

        d = df.copy()
        for c in V2_ALL_COLS:
            if c not in d.columns:
                d[c] = ""
        d = d.reindex(columns=V2_ALL_COLS, fill_value="")
        parts.append(d)

    if not parts:
        hist = pd.DataFrame(columns=V2_ALL_COLS)
    else:
        hist = pd.concat(parts, ignore_index=True)
        hist["trade_date"] = hist["trade_date"].astype(str)
        hist["rank"] = pd.to_numeric(hist["rank"], errors="coerce")
        hist = hist.reindex(columns=V2_ALL_COLS, fill_value="")
        hist = hist.sort_values(
            by=["trade_date", "rank", "ts_code"],
            ascending=[True, True, True],
            na_position="last",
        )
        hist = hist.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)

    if skipped:
        print(f"[HISTORY] skipped non-V2 files: {', '.join(skipped)}")
    print(f"[HISTORY] rebuild rows={len(hist)}")
    return hist


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    outputs_dir = repo_root / "outputs"
    learning_dir = outputs_dir / "learning"

    learning_dir.mkdir(parents=True, exist_ok=True)

    trade_dates: List[str] = []

    for p in sorted(outputs_dir.glob("predict_top10_*.json")):
        m = re.match(r"predict_top10_(\d{8})\.json$", p.name)
        if m:
            trade_dates.append(m.group(1))

    for p in sorted(learning_dir.glob("pred_top10_*.csv")):
        m = re.match(r"pred_top10_(\d{8})\.csv$", p.name)
        if m:
            trade_dates.append(m.group(1))

    trade_dates = sorted(set(trade_dates))
    print(f"[SCAN] trade_dates={len(trade_dates)}")

    for td in trade_dates:
        _migrate_one_day(outputs_dir, learning_dir, td)

    hist = _rebuild_history_from_daily(learning_dir)
    hist_path = learning_dir / "pred_top10_history.csv"
    hist.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {hist_path} rows={len(hist)}")


if __name__ == "__main__":
    main()
