from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd


CODE_COL_CANDIDATES = [
    "ts_code",
    "code",
    "TS_CODE",
    "证券代码",
    "股票代码",
]
NAME_COL_CANDIDATES = ["name", "stock_name", "名称"]
PROB_COL_CANDIDATES = ["prob", "Probability", "概率", "涨停概率"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业"]


def _df_to_md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    if df is None or df.empty:
        return ""

    d = df
    if cols is not None:
        use_cols = [c for c in cols if c in d.columns]
        if use_cols:
            d = d[use_cols].copy()

    col_map = {
        "rank": "排名",
        "ts_code": "代码",
        "name": "股票",
        "score": "Score",
        "prob": "Probability",
        "StrengthScore": "强度得分",
        "ThemeBoost": "题材加成",
        "board": "板块",
    }
    d = d.rename(columns=col_map)

    try:
        return d.to_markdown(index=False)
    except Exception:
        x = d.copy().fillna("")
        headers = list(x.columns)

        def esc(v: Any) -> str:
            s = str(v)
            s = s.replace("\n", " ").replace("\r", " ")
            s = s.replace("|", "\\|")
            return s

        lines = []
        lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for _, row in x.iterrows():
            lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
        return "\n".join(lines)


def _pick_first_not_none(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            v = d.get(k)
            if v is not None:
                return v
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


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _norm_code(code: Any) -> Tuple[str, str]:
    s = _safe_str(code).upper()
    if not s:
        return "", ""
    code6 = s.split(".")[0]
    if len(code6) > 6 and code6.isdigit():
        code6 = code6[-6:]
    return s, code6


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
            p = Path(v)
            return p
        except Exception:
            continue
    return None


def _read_csv_guess(p: Path) -> pd.DataFrame:
    if p is None or not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


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
    df = _ctx_df(ctx, ["limit_df", "limit_list", "limit", "limit_list_d"])
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


def _topN_to_hit_df(topN_df: Optional[pd.DataFrame], limit_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    if topN_df is None or topN_df.empty:
        metrics.update({"hit_count": 0, "top_count": 0, "limit_count": 0, "hit_rate": ""})
        return pd.DataFrame(), metrics

    df = topN_df.copy()

    code_col = _first_existing_col(df, CODE_COL_CANDIDATES)
    if not code_col:
        metrics.update({"hit_count": 0, "top_count": len(df), "limit_count": 0, "hit_rate": ""})
        df["命中"] = ""
        df["板块"] = df.get("board", "")
        return df, metrics

    if code_col != "ts_code":
        df["ts_code"] = df[code_col]

    prob_col = _first_existing_col(df, PROB_COL_CANDIDATES)
    if prob_col and prob_col != "prob":
        df["prob"] = df[prob_col]

    name_col = _first_existing_col(df, NAME_COL_CANDIDATES)
    if name_col and name_col != "name":
        df["name"] = df[name_col]

    board_col = _first_existing_col(df, BOARD_COL_CANDIDATES)
    if board_col and board_col != "板块":
        df["板块"] = df[board_col].fillna("")
    elif "板块" not in df.columns:
        df["板块"] = df.get("board", "").fillna("")

    top_count = len(df)

    _, limit_ts, limit_c6 = _build_code_sets(limit_df, CODE_COL_CANDIDATES)
    limit_count = max(len(limit_ts), len(limit_c6))

    hits = []
    hit_count = 0
    for v in df["ts_code"].tolist():
        ts, c6 = _norm_code(v)
        ok = (ts in limit_ts) or (c6 in limit_c6)
        hits.append("是" if ok else "否")
        if ok:
            hit_count += 1

    df["命中"] = hits

    rate = 0.0
    if top_count > 0:
        rate = hit_count * 100.0 / float(top_count)
    metrics.update({
        "hit_count": int(hit_count),
        "top_count": int(top_count),
        "limit_count": int(limit_count),
        "hit_rate": f"{round(rate)}%" if top_count > 0 else "",
    })
    return df, metrics


def _recent_hit_history(outdir: Path, settings, ctx, max_days: int = 10) -> pd.DataFrame:
    files = sorted(outdir.glob("predict_top10_*.json"), key=lambda p: p.name, reverse=True)
    rows: List[Dict[str, Any]] = []
    for f in files:
        m = re.match(r"predict_top10_(\d{8})\.json", f.name)
        if not m:
            continue
        d = m.group(1)
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        metrics = payload.get("metrics") if isinstance(payload, dict) else None

        hit_count = None
        top_count = None
        limit_count = None
        hit_rate = None

        if isinstance(metrics, dict):
            hit_count = metrics.get("hit_count")
            top_count = metrics.get("top_count")
            limit_count = metrics.get("limit_count")
            hit_rate = metrics.get("hit_rate")

        if hit_count is None or top_count is None:
            top = payload.get("topN") or payload.get("topn") or []
            tdf = _to_df(top)
            ldf = _load_limit_df(settings, ctx, d)
            hit_df, mres = _topN_to_hit_df(tdf, ldf)
            hit_count = mres.get("hit_count")
            top_count = mres.get("top_count")
            limit_count = mres.get("limit_count")
            hit_rate = mres.get("hit_rate")

        if hit_count is None or top_count is None:
            continue
        rows.append({
            "日期": d,
            "命中数": int(hit_count),
            "命中率": str(hit_rate) if hit_rate is not None else "",
            "当日涨停家数": int(limit_count) if limit_count is not None else 0,
        })
        if len(rows) >= max_days:
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn) -> None:
    # outputs_dir
    outdir = getattr(getattr(settings, "io", None), "outputs_dir", None)
    if not outdir:
        outdir = "outputs"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # parse topn
    topN_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        topN_df = _pick_first_not_none(topn, ["topN", "topn", "TopN", "top"])
        full_df = topn.get("full") if "full" in topn else None
    else:
        topN_df = topn

    topN_df = _to_df(topN_df)
    full_df = _to_df(full_df)

    # compute hit
    limit_df = _load_limit_df(settings, ctx, trade_date)
    hit_df, metrics = _topN_to_hit_df(topN_df, limit_df)

    # JSON
    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "gate": gate,
        "topN": [] if topN_df is None else topN_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics,
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    # Markdown
    md_path = outdir / f"predict_top10_{trade_date}.md"
    lines = [f"# Top10 Prediction ({trade_date})\n"]

    if topN_df is None or topN_df.empty:
        reason = ""
        try:
            if isinstance(gate, dict):
                r = gate.get("reason") or gate.get("msg") or ""
                if r:
                    reason = f"（{r}）"
        except Exception:
            pass
        lines.append(f"⚠️ Gate 未通过，Top10 为空。{reason}\n")
    else:
        lines.append("## ✅ 收盘命中检查\n")
        lines.append(_df_to_md_table(hit_df, cols=["ts_code", "name", "prob", "命中", "板块"]))
        lines.append("\n")

    # limit list
    lines.append("### 当日涨停标的（limit_list）\n")
    if limit_df is None or limit_df.empty:
        lines.append("(未能找到 limit_list_d.csv)\n\n")
    else:
        code_col = _first_existing_col(limit_df, CODE_COL_CANDIDATES) or "ts_code"
        show = limit_df[[code_col]].head(30).copy()
        lines.append(_df_to_md_table(show, cols=[code_col]))
        lines.append("\n")

    # history
    hist_df = _recent_hit_history(outdir, settings, ctx, max_days=10)
    if hist_df is not None and not hist_df.empty:
        lines.append("## 📈 最近10日 Top10 命中率\n")
        lines.append(_df_to_md_table(hist_df, cols=["日期", "命中数", "命中率", "当日涨停家数"]))
        lines.append("\n")

    # full ranking
    if full_df is not None and not full_df.empty:
        lines.append("## 📊 Full Ranking (All Candidates After Step6)\n")
        full_sorted = full_df.copy()
        try:
            if "_score" in full_sorted.columns:
                full_sorted = full_sorted.sort_values(
                    by=["_score", "_prob"] if "_prob" in full_sorted.columns else ["_score"],
                    ascending=False,
                )
            elif "score" in full_sorted.columns:
                full_sorted = full_sorted.sort_values(
                    by=["score", "prob"] if "prob" in full_sorted.columns else ["score"],
                    ascending=False,
                )
            elif "prob" in full_sorted.columns:
                full_sorted = full_sorted.sort_values(by=["prob"], ascending=False)
        except Exception:
            pass

        full_sorted = full_sorted.head(50)
        display_cols = [
            "rank",
            "ts_code",
            "name",
            "score",
            "prob",
            "StrengthScore",
            "ThemeBoost",
            "board",
        ]
        lines.append(_df_to_md_table(full_sorted, cols=display_cols))
        lines.append("\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"✅ Outputs written: {md_path}")
