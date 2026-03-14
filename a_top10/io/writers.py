from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


CODE_COL_CANDIDATES = ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"]
NAME_COL_CANDIDATES = ["name", "stock_name", "名称", "股票", "证券名称", "股票简称"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"]

# =========================
# V2 canonical schema
# =========================
V2_BASE_COLS = [
    "rank",
    "ts_code",
    "name",
    "prob_rule",
    "prob_ml",
    "prob_final",
    "prob",         # thin output alias of prob_final
    "Probability",  # thin output alias of prob_final
    "final_score",
    "score",        # thin output alias of final_score
    "StrengthScore",
    "ThemeBoost",
    "board",
]
V2_META_COLS = ["trade_date", "verify_date", "run_id", "run_attempt", "commit_sha", "generated_at_utc"]
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


# =========================
# basic utils
# =========================
def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_run_meta() -> Dict[str, str]:
    run_id = (os.getenv("RUN_ID") or os.getenv("GITHUB_RUN_ID") or "").strip()
    run_attempt = (os.getenv("RUN_ATTEMPT") or os.getenv("GITHUB_RUN_ATTEMPT") or "").strip()
    sha = (os.getenv("COMMIT_SHA") or os.getenv("GITHUB_SHA") or "").strip()

    if not run_id:
        run_id = datetime.utcnow().strftime("ts%Y%m%d%H%M%S")
    if not run_attempt:
        run_attempt = "1"

    return {
        "run_id": run_id,
        "run_attempt": run_attempt,
        "commit_sha": sha,
        "generated_at_utc": _utc_now_iso(),
    }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_df(x: Any) -> Optional[pd.DataFrame]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    try:
        return pd.DataFrame(x)
    except Exception:
        return None


def _json_default(o: Any) -> str:
    try:
        return str(o)
    except Exception:
        return repr(o)


def _read_csv_guess(p: Path) -> pd.DataFrame:
    if p is None or not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _write_text_once(path: Path, text: str, *, force: bool = False, encoding: str = "utf-8") -> bool:
    _ensure_dir(path.parent)
    if path.exists() and not force:
        print(f"[SKIP] exists (write-once): {path}")
        return False
    path.write_text(text, encoding=encoding)
    print(f"[WRITE] {path}")
    return True


def _write_text_overwrite(path: Path, text: str, *, encoding: str = "utf-8") -> bool:
    _ensure_dir(path.parent)
    path.write_text(text, encoding=encoding)
    print(f"[WRITE] {path} (overwrite)")
    return True


def _write_csv_once(df: pd.DataFrame, path: Path, *, force: bool = False) -> bool:
    _ensure_dir(path.parent)
    if path.exists() and not force:
        print(f"[SKIP] exists (write-once): {path}")
        return False
    if df is None:
        df = pd.DataFrame(columns=V2_ALL_COLS)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path} rows={len(df)}")
    return True


def _write_csv_overwrite(df: pd.DataFrame, path: Path) -> bool:
    _ensure_dir(path.parent)
    if df is None:
        df = pd.DataFrame(columns=V2_ALL_COLS)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path} rows={len(df)} (overwrite)")
    return True


# =========================
# market / calendar helpers
# =========================
def _ctx_df(ctx: Any, keys: Sequence[str]) -> Optional[pd.DataFrame]:
    if not isinstance(ctx, dict):
        return None
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


def _ctx_path(ctx: Any, keys: Sequence[str]) -> Optional[Path]:
    if not isinstance(ctx, dict):
        return None
    for k in keys:
        v = ctx.get(k)
        if v is None:
            continue
        try:
            return Path(v)
        except Exception:
            continue
    return None


def _ctx_trade_date(ctx: Any) -> str:
    if not isinstance(ctx, dict):
        return ""
    for k in ("trade_date", "TRADE_DATE", "asof", "date", "snapshot_date"):
        v = ctx.get(k)
        s = _safe_str(v)
        if len(s) == 8 and s.isdigit():
            return s
    return ""


def _norm_code(code: Any) -> Tuple[str, str]:
    s = _safe_str(code).upper()
    if not s:
        return "", ""
    code6 = s.split(".")[0]
    if len(code6) > 6 and code6.isdigit():
        code6 = code6[-6:]
    return s, code6


def _build_code_sets(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[str, set, set]:
    ts_set: set = set()
    c6_set: set = set()
    code_col = _first_existing_col(df, candidates)
    if code_col:
        for v in df[code_col].tolist():
            ts, c6 = _norm_code(v)
            if ts:
                ts_set.add(ts)
            if c6:
                c6_set.add(c6)
    return code_col or "", ts_set, c6_set


def _count_limitups(limit_df: Optional[pd.DataFrame]) -> int:
    if limit_df is None or limit_df.empty:
        return 0

    code_col = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not code_col:
        return int(len(limit_df))

    uniq: set = set()
    for v in limit_df[code_col].tolist():
        ts, c6 = _norm_code(v)
        if ts:
            uniq.add(ts)
        elif c6:
            uniq.add(c6)

    return int(len(uniq)) if uniq else int(len(limit_df))


def _resolve_snapshot_dir(settings, ctx, trade_date: str) -> Optional[Path]:
    cands: List[Path] = []

    p = _ctx_path(ctx, ["snapshot_dir", "snap_dir", "snapshot_path"])
    if p:
        cands.append(p)

    sio = getattr(settings, "io", None)
    if sio is not None:
        for attr in ("snapshot_dir", "snapshots_dir", "snapshot_root", "snapshot_base", "snapshots_root"):
            v = getattr(sio, attr, None)
            if v:
                try:
                    cands.append(Path(v))
                except Exception:
                    pass

    dr = getattr(settings, "data_repo", None)
    if dr is not None:
        for attr in ("snapshot_dir", "snapshot_path", "get_snapshot_dir"):
            fn = getattr(dr, attr, None)
            if callable(fn):
                try:
                    cands.append(Path(fn(trade_date)))
                except Exception:
                    pass
        for attr in ("root", "warehouse_root"):
            v = getattr(dr, attr, None)
            if v:
                try:
                    cands.append(Path(v))
                except Exception:
                    pass

    for base in cands:
        try:
            b = Path(base)
        except Exception:
            continue
        if not b.exists():
            continue
        if trade_date in str(b):
            return b
        if (b / trade_date).exists():
            return b / trade_date
        if len(trade_date) >= 8:
            y = trade_date[:4]
            for cand in (
                b / y / trade_date,
                b / "snapshots" / trade_date,
                b / "data" / "raw" / y / trade_date,
                b / "raw" / y / trade_date,
            ):
                if cand.exists():
                    return cand

    for guess in (Path("data_repo/snapshots"), Path("data_repo"), Path("snapshots"), Path("_warehouse")):
        if (guess / trade_date).exists():
            return guess / trade_date

    return None


def _load_limit_df(settings, ctx, trade_date: str) -> pd.DataFrame:
    ctx_td = _ctx_trade_date(ctx)
    if ctx_td and ctx_td == trade_date:
        df = _ctx_df(ctx, ["limit_df", "limit_list", "limit", "limit_list_d", "limit_up", "limitup"])
        if df is not None and not df.empty:
            return df.copy()

    snap = _resolve_snapshot_dir(settings, ctx, trade_date)
    if snap is None:
        return pd.DataFrame()

    for fn in ("limit_list_d.csv", "limit_list.csv", "limit_listd.csv", "limit_list_d"):
        p = snap / fn
        if p.exists():
            return _read_csv_guess(p)
    return pd.DataFrame()


def _list_trade_dates(settings) -> List[str]:
    dates: List[str] = []
    dr = getattr(settings, "data_repo", None)
    if dr is not None:
        fn = getattr(dr, "list_snapshot_dates", None)
        if callable(fn):
            try:
                dates = list(fn()) or []
            except Exception:
                pass
    return sorted([d for d in dates if isinstance(d, str) and len(d) == 8 and d.isdigit()])


def _prev_next_trade_date(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    if calendar and trade_date in calendar:
        i = calendar.index(trade_date)
        prev_td = calendar[i - 1] if i - 1 >= 0 else ""
        next_td = calendar[i + 1] if i + 1 < len(calendar) else ""
        return prev_td, next_td
    return "", ""


def _prev_next_trade_date_with_fallback(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    prev_td, next_td = _prev_next_trade_date(calendar, trade_date)
    if prev_td and next_td:
        return prev_td, next_td

    try:
        d = datetime.strptime(trade_date, "%Y%m%d")
    except Exception:
        return prev_td or "", next_td or ""

    prev_d = d - timedelta(days=1)
    while prev_d.weekday() >= 5:
        prev_d -= timedelta(days=1)

    next_d = d + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)

    if not prev_td:
        prev_td = prev_d.strftime("%Y%m%d")
    if not next_td:
        next_td = next_d.strftime("%Y%m%d")

    return prev_td, next_td


# =========================
# V2 canonicalization
# =========================
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
    """
    V2 唯一标准化入口。
    核心主语义只认：
    - prob_final
    - final_score
    其它别名仅在产物层镜像，不参与核心语义竞争。
    """
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

    if "prob_final" in src.columns:
        out["prob_final"] = src["prob_final"]
    else:
        out["prob_final"] = ""

    if "final_score" in src.columns:
        out["final_score"] = src["final_score"]
    else:
        out["final_score"] = ""

    out["StrengthScore"] = src["StrengthScore"] if "StrengthScore" in src.columns else (
        src["强度得分"] if "强度得分" in src.columns else ""
    )
    out["ThemeBoost"] = src["ThemeBoost"] if "ThemeBoost" in src.columns else (
        src["题材加成"] if "题材加成" in src.columns else ""
    )

    for col in ["prob_rule", "prob_ml", "prob_final", "final_score", "StrengthScore", "ThemeBoost"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # thin output aliases only
    out["prob"] = out["prob_final"]
    out["Probability"] = out["prob_final"]
    out["score"] = out["final_score"]

    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out = out.reindex(columns=V2_BASE_COLS, fill_value="")
    return out


def _enrich_v2_with_meta(df: Optional[pd.DataFrame], trade_date: str, verify_date: str, run_meta: Dict[str, str]) -> pd.DataFrame:
    base = _canonicalize_v2_frame(df)
    out = base.copy()
    out.insert(0, "trade_date", trade_date)
    out.insert(1, "verify_date", verify_date)
    out["run_id"] = run_meta["run_id"]
    out["run_attempt"] = run_meta["run_attempt"]
    out["commit_sha"] = run_meta["commit_sha"]
    out["generated_at_utc"] = run_meta["generated_at_utc"]
    out = out.reindex(columns=V2_ALL_COLS, fill_value="")
    return out


# =========================
# markdown helpers
# =========================
def _df_to_md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    if df is None or df.empty:
        return ""

    d = df.copy()
    if cols is not None:
        use_cols = [c for c in cols if c in d.columns]
        if use_cols:
            d = d[use_cols].copy()

    col_map = {
        "rank": "排名",
        "rank_limit": "排名",
        "ts_code": "代码",
        "name": "股票",
        "prob_final": "prob_final",
        "final_score": "final_score",
        "StrengthScore": "强度得分",
        "ThemeBoost": "题材加成",
        "board": "板块",
        "命中": "命中",
        "日期": "日期",
        "命中数": "命中数",
        "命中率": "命中率",
        "当日涨停家数": "当日涨停家数",
        "Probability": "Probability",
    }
    d = d.rename(columns=col_map)

    try:
        return d.to_markdown(index=False)
    except Exception:
        x = d.copy().fillna("")
        headers = list(x.columns)

        def esc(v: Any) -> str:
            s = str(v)
            s = s.replace("\n", " ").replace("\r", " ").replace("|", "\\|")
            return s

        lines = []
        lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for _, row in x.iterrows():
            lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
        return "\n".join(lines)


# =========================
# report helpers
# =========================
def _load_json_topn(outdir: Path, td: str) -> Optional[pd.DataFrame]:
    if not td:
        return None
    p = outdir / f"predict_top10_{td}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw_topn = payload.get("topN")
    if raw_topn is None:
        raw_topn = []
    return _to_df(raw_topn)


def _topn_to_hit_df(topn_df: Optional[pd.DataFrame], limit_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    v2 = _canonicalize_v2_frame(topn_df)

    if v2.empty:
        metrics.update({"hit_count": 0, "top_count": 0, "limit_count": _count_limitups(limit_df), "hit_rate": ""})
        return pd.DataFrame(), metrics

    _, limit_ts, limit_c6 = _build_code_sets(limit_df, CODE_COL_CANDIDATES)

    hits: List[str] = []
    hit_count = 0
    for v in v2["ts_code"].tolist():
        ts, c6 = _norm_code(v)
        ok = (ts in limit_ts) or (c6 in limit_c6)
        hits.append("是" if ok else "否")
        if ok:
            hit_count += 1

    out = v2.copy()
    out["命中"] = hits

    top_count = int(len(out))
    limit_count = int(_count_limitups(limit_df))
    hit_rate = f"{round(hit_count * 100.0 / float(top_count))}%" if top_count > 0 else ""

    metrics.update({
        "hit_count": int(hit_count),
        "top_count": top_count,
        "limit_count": limit_count,
        "hit_rate": hit_rate,
    })
    return out, metrics


def _standardize_strength_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if "排名" not in d.columns:
        if "rank_limit" in d.columns:
            d["排名"] = d["rank_limit"]
        elif "rank" in d.columns:
            d["排名"] = d["rank"]
        else:
            d["排名"] = range(1, len(d) + 1)

    if "代码" not in d.columns:
        c = _first_existing_col(d, CODE_COL_CANDIDATES)
        d["代码"] = d[c] if c else ""

    if "股票" not in d.columns:
        n = _first_existing_col(d, NAME_COL_CANDIDATES)
        d["股票"] = d[n] if n else ""

    if "Probability" not in d.columns:
        if "prob_final" in d.columns:
            d["Probability"] = d["prob_final"]
        else:
            d["Probability"] = ""

    if "强度得分" not in d.columns:
        if "StrengthScore" in d.columns:
            d["强度得分"] = d["StrengthScore"]
        else:
            d["强度得分"] = ""

    if "题材加成" not in d.columns:
        if "ThemeBoost" in d.columns:
            d["题材加成"] = d["ThemeBoost"]
        else:
            d["题材加成"] = ""

    if "板块" not in d.columns:
        b = _first_existing_col(d, BOARD_COL_CANDIDATES)
        d["板块"] = d[b] if b else ""

    out = d[["排名", "代码", "股票", "Probability", "强度得分", "题材加成", "板块"]].copy()
    out["强度得分"] = pd.to_numeric(out["强度得分"], errors="coerce")
    out = out.sort_values(by="强度得分", ascending=False, na_position="last").reset_index(drop=True)
    out["排名"] = range(1, len(out) + 1)
    return out


def _join_limit_strength(limit_df: pd.DataFrame, full_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if limit_df is None or limit_df.empty:
        return None

    lcode = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not lcode:
        return None

    l = limit_df[[lcode]].copy()
    if lcode != "ts_code":
        l["ts_code"] = l[lcode]

    full_v2 = _canonicalize_v2_frame(full_df)
    if full_v2.empty:
        out = l[["ts_code"]].copy()
        out["name"] = ""
        out["prob_final"] = ""
        out["StrengthScore"] = ""
        out["ThemeBoost"] = ""
        out["board"] = ""
        out.insert(0, "rank_limit", range(1, len(out) + 1))
        return out

    merged = pd.merge(l[["ts_code"]], full_v2, on="ts_code", how="left")
    merged = merged.sort_values(by=["StrengthScore", "final_score", "prob_final"], ascending=False, na_position="last")
    merged = merged.reset_index(drop=True)
    merged.insert(0, "rank_limit", range(1, len(merged) + 1))
    return merged


def _recent_hit_history(outdir: Path, settings, ctx, max_days: int = 10) -> pd.DataFrame:
    calendar = _list_trade_dates(settings)
    files = sorted(outdir.glob("predict_top10_*.json"), key=lambda p: p.name, reverse=True)
    rows: List[Dict[str, Any]] = []

    for f in files:
        m = re.match(r"predict_top10_(\d{8})\.json", f.name)
        if not m:
            continue
        pred_date = m.group(1)

        _, verify_date = _prev_next_trade_date(calendar, pred_date)
        if not verify_date:
            continue

        topn_df = _load_json_topn(outdir, pred_date)
        limit_df_verify = _load_limit_df(settings, ctx, verify_date)
        _, metrics = _topn_to_hit_df(topn_df, limit_df_verify)

        rows.append({
            "日期": pred_date,
            "命中数": int(metrics.get("hit_count") or 0),
            "命中率": str(metrics.get("hit_rate") or ""),
            "当日涨停家数": _count_limitups(limit_df_verify),
        })
        if len(rows) >= max_days:
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =========================
# V2 history rebuild
# =========================
def _is_v2_compatible_schema(df: pd.DataFrame) -> bool:
    cols = set(str(c).strip() for c in df.columns)
    return V2_REQUIRED_HISTORY_COLS.issubset(cols)


def _rebuild_v2_history(learning_dir: Path) -> pd.DataFrame:
    """
    只从 V2 的 pred_top10_YYYYMMDD.csv 日文件重建 history。
    判定标准改为“V2 兼容 schema”：
    - 必需列存在即可
    - 允许额外列存在
    - 允许列顺序不同
    """
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
        hist = hist.reindex(columns=V2_ALL_COLS, fill_value="")
        hist["trade_date"] = hist["trade_date"].astype(str)
        hist["rank"] = pd.to_numeric(hist["rank"], errors="coerce")
        hist = hist.sort_values(by=["trade_date", "rank", "ts_code"], ascending=[True, True, True], na_position="last")
        hist = hist.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)

    if skipped:
        print(f"[HISTORY] skipped non-V2 daily files: {', '.join(skipped)}")
    print(f"[HISTORY] rebuild rows={len(hist)}")
    return hist


# =========================
# main writer
# =========================
def write_outputs(settings, trade_date: str, ctx, gate, topn, learn) -> None:
    outdir = getattr(getattr(settings, "io", None), "outputs_dir", None) or "outputs"
    outdir = Path(outdir)
    _ensure_dir(outdir)

    run_meta = _get_run_meta()

    learning_dir = outdir / "learning"
    warehouse_dir = outdir / "_warehouse" / "pred_top10"
    decisio_dir = outdir / "decisio"

    _ensure_dir(learning_dir)
    _ensure_dir(warehouse_dir)
    _ensure_dir(decisio_dir)

    topn_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None
    limit_up_table_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        raw_topn = None
        for k in ["topN", "topn", "TopN", "top"]:
            if k in topn and topn.get(k) is not None:
                raw_topn = topn.get(k)
                break

        topn_df = _to_df(raw_topn)

        raw_full = topn.get("full") if "full" in topn else None
        full_df = _to_df(raw_full)

        raw_limit_up_table = topn.get("limit_up_table") if "limit_up_table" in topn else None
        limit_up_table_df = _to_df(raw_limit_up_table)
    else:
        topn_df = _to_df(topn)

    calendar = _list_trade_dates(settings)
    prev_td, next_td = _prev_next_trade_date_with_fallback(calendar, trade_date)

    limit_df_current = _load_limit_df(settings, ctx, trade_date)
    _, metrics_same_day = _topn_to_hit_df(topn_df, limit_df_current)

    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "verify_date": next_td,
        "gate": gate,
        "topN": [] if topn_df is None else topn_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "limit_up_table": [] if limit_up_table_df is None else limit_up_table_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics_same_day,
        "run_meta": run_meta,
    }

    # -----------------
    # JSON / MD
    # -----------------
    # V2 壳层收口：
    # dated json/md 不能再 write-once，否则同一 trade_date rerun 后
    # CSV 已更新而 md/json 不更新，会制造“同日多真相”。
    json_path = outdir / f"predict_top10_{trade_date}.json"
    _write_text_overwrite(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    md_lines: List[str] = [f"# {trade_date} 预测报告\n"]
    md_lines.append(f"## 《{trade_date} 预测：{next_td} 涨停 TOP 10》\n")

    topn_v2 = _canonicalize_v2_frame(topn_df)
    if topn_v2.empty:
        reason = ""
        if isinstance(gate, dict):
            msg = _safe_str(gate.get("reason") or gate.get("msg") or "")
            if msg:
                reason = f"（{msg}）"
        md_lines.append(f"⚠️ Gate 未通过，Top10 为空。{reason}\n")
    else:
        md_lines.append(
            _df_to_md_table(
                topn_v2,
                cols=["rank", "ts_code", "name", "prob_final", "final_score", "StrengthScore", "ThemeBoost", "board"],
            )
        )
        md_lines.append("")

    md_lines.append(f"## 《{trade_date} 所有涨停股票的强度列表》\n")
    strength_limit_df = None
    if limit_up_table_df is not None and not limit_up_table_df.empty:
        strength_limit_df = _standardize_strength_table(limit_up_table_df)
    else:
        joined = _join_limit_strength(limit_df_current, full_df)
        if joined is not None and not joined.empty:
            strength_limit_df = _standardize_strength_table(joined)

    if strength_limit_df is None or strength_limit_df.empty:
        md_lines.append("(未能生成强度列表：limit_list 或 full 排名为空)\n")
    else:
        md_lines.append(
            _df_to_md_table(
                strength_limit_df,
                cols=["排名", "代码", "股票", "Probability", "强度得分", "题材加成", "板块"],
            )
        )
        md_lines.append("")

    prev_title = prev_td if prev_td else "上一交易日"
    md_lines.append(f"## 《{prev_title} 预测：{trade_date} 命中情况》\n")
    prev_topn_df = _load_json_topn(outdir, prev_td)
    prev_hit_df, _prev_metrics = _topn_to_hit_df(prev_topn_df, limit_df_current)
    if prev_hit_df.empty:
        md_lines.append("（未找到上一交易日预测文件或上一交易日 Top10 为空）\n")
    else:
        md_lines.append(_df_to_md_table(prev_hit_df, cols=["ts_code", "name", "prob_final", "命中", "board"]))
        md_lines.append("")

    hist10 = _recent_hit_history(outdir, settings, ctx, max_days=10)
    if not hist10.empty:
        md_lines.append("## 《近10日 Top10 命中率》\n")
        md_lines.append(_df_to_md_table(hist10, cols=["日期", "命中数", "命中率", "当日涨停家数"]))
        md_lines.append("")

    md_text = "\n".join(md_lines)
    md_path = outdir / f"predict_top10_{trade_date}.md"
    _write_text_overwrite(md_path, md_text, encoding="utf-8")

    latest_md = outdir / "latest.md"
    latest_md.write_text(md_text, encoding="utf-8")
    print(f"[WRITE] {latest_md} (latest)")

    # -----------------
    # V2 CSV outputs
    # -----------------
    topn_out = _enrich_v2_with_meta(topn_df, trade_date, next_td, run_meta)
    full_out = _enrich_v2_with_meta(full_df, trade_date, next_td, run_meta)

    # learning daily / latest / warehouse
    # 这里必须直接覆盖写：
    # pred_top10_{trade_date}.csv 现在是 V2 history 的正式源文件，
    # 不能再被旧文件钉死。
    learn_csv = learning_dir / f"pred_top10_{trade_date}.csv"
    _write_csv_overwrite(topn_out, learn_csv)

    learn_latest = learning_dir / "pred_top10_latest.csv"
    _write_csv_overwrite(topn_out, learn_latest)

    # warehouse 仍保留归档语义：write-once
    wh_csv = warehouse_dir / f"pred_top10_{trade_date}_{run_meta['run_id']}.csv"
    _write_csv_once(topn_out, wh_csv, force=False)

    # history: V2 rebuild only
    hist_path = learning_dir / "pred_top10_history.csv"
    hist_df = _rebuild_v2_history(learning_dir)
    _write_csv_overwrite(hist_df, hist_path)

    # decisio: same old path / file names, V2 schema inside
    # dated decisio 也要覆盖写，避免 rerun 后 latest 已更新但 dated 旧壳残留
    decisio_dated = decisio_dir / f"pred_decisio_{trade_date}.csv"
    _write_csv_overwrite(full_out, decisio_dated)

    decisio_latest = decisio_dir / "pred_decisio_latest.csv"
    _write_csv_overwrite(full_out, decisio_latest)

    # last run
    last_run = learning_dir / "_last_run.txt"
    last_run.write_text(
        (
            f"trade_date={trade_date}\n"
            f"verify_date={next_td}\n"
            f"run_id={run_meta['run_id']}\n"
            f"run_attempt={run_meta['run_attempt']}\n"
            f"commit_sha={run_meta['commit_sha']}\n"
            f"generated_at_utc={run_meta['generated_at_utc']}\n"
        ),
        encoding="utf-8",
    )
    print(f"[WRITE] {last_run} (latest)")

    print(f"✅ V2 outputs written: {md_path}")
