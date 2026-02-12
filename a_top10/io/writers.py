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


def _list_trade_dates(settings) -> List[str]:
    """
    用 DataRepo 自带 list_snapshot_dates 作为交易日历（最稳）。
    找不到就兜底用已有 outputs 文件名推断。
    """
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

    # 兜底：不懂交易日历就返回空（不要乱加日期，避免误导）
    return "", ""


def _load_json_topN(outdir: Path, td: str) -> Optional[pd.DataFrame]:
    p = outdir / f"predict_top10_{td}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    top = payload.get("topN") or payload.get("topn") or []
    return _to_df(top)


def _join_limit_strength(limit_df: pd.DataFrame, full_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    筛出当日涨停股，并输出 StrengthScore / ThemeBoost / board（强度列表）
    都来自主程序字段，writer 只做 join/filter。
    """
    if limit_df is None or limit_df.empty:
        return None

    lcode = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not lcode:
        return None

    # 如果 full_df 不存在，尽量用 limit_df 自身输出
    if full_df is None or full_df.empty:
        df = limit_df[[lcode]].copy()
        df["板块"] = ""
        return df.rename(columns={lcode: "ts_code"})

    fcode = _first_existing_col(full_df, CODE_COL_CANDIDATES)
    if not fcode:
        return None

    l = limit_df[[lcode]].copy()
    f = full_df.copy()

    if lcode != "ts_code":
        l["ts_code"] = l[lcode]
    if fcode != "ts_code":
        f["ts_code"] = f[fcode]

    # 优先用 StrengthScore 作为强度排序，其次 score 再其次 prob
    sort_cols = []
    if "StrengthScore" in f.columns:
        sort_cols = ["StrengthScore"]
    elif "score" in f.columns:
        sort_cols = ["score"]
    elif "prob" in f.columns:
        sort_cols = ["prob"]

    # merge
    m = pd.merge(l[["ts_code"]], f, on="ts_code", how="left")

    # 结果输出必要列
    # 保证 name 列存在（给 _df_to_md_table 重命名）
    if "name" not in m.columns and _first_existing_col(m, NAME_COL_CANDIDATES):
        m["name"] = m[_first_existing_col(m, NAME_COL_CANDIDATES)]
    if "板块" not in m.columns and _first_existing_col(m, BOARD_COL_CANDIDATES):
        m["板块"] = m[_first_existing_col(m, BOARD_COL_CANDIDATES)]

    if sort_cols:
        m = m.sort_values(by=sort_cols, ascending=False)

    return m


def _recent_hit_history(outdir: Path, settings, ctx, max_days: int = 10) -> pd.DataFrame:
    """
    近10日命中率口径：predict_date -> next_trade_date 的 limit_list。
    """
    calendar = _list_trade_dates(settings)
    files = sorted(outdir.glob("predict_top10_*.json"), key=lambda p: p.name, reverse=True)
    rows: List[Dict[str, Any]] = []

    for f in files:
        m = re.match(r"predict_top10_(\d{8})\.json", f.name)
        if not m:
            continue
        d = m.group(1)

        _, next_td = _prev_next_trade_date(calendar, d)
        if not next_td:
            # 如果没有 next_td，无法验证命中，就跳过
            continue

        topN_df = _load_json_topN(outdir, d)
        ldf = _load_limit_df(settings, ctx, next_td)
        hit_df, mres = _topN_to_hit_df(topN_df, ldf)

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

    # trade calendar
    calendar = _list_trade_dates(settings)
    prev_td, next_td = _prev_next_trade_date(calendar, trade_date)

    # 本日 limit_list（用于第2块验证命中；第3块强度列表）
    limit_df_current = _load_limit_df(settings, ctx, trade_date)

    # JSON payload：保持兼容，不大动 payload 结构（metrics 仍按“预测日验证预测日”的旧口径）
    hit_df_same_day, metrics_same_day = _topN_to_hit_df(topN_df, limit_df_current)

    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "verify_date": next_td,  # 下一交易日：预测目标日（report 第1块标题里的那个）
        "gate": gate,
        "topN": [] if topN_df is None else topN_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics_same_day,  # 保持原字段用途
        "metrics_same_day": metrics_same_day,
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    # Markdown（你要的四块中文标题）
    md_path = outdir / f"predict_top10_{trade_date}.md"
    lines = [f"# {trade_date} 预测报告\n"]

    # 第1块：预测（trade_date -> next_td）
    if next_td:
        lines.append(f"## 《{trade_date} 预测：{next_td} 涨停 TOP 10》\n")
    else:
        lines.append(f"## 《{trade_date} 预测：下一交易日 涨停 TOP 10》\n")

    if topN_df is None or topN_df.empty:
        reason = ""
        try:
            if isinstance(gate, dict):
                r = gate.get("reason") or gate.get("msg") or ""
                if r:
                    reason = f"（{r}）"
        except Exception:
            pass
        lines.append(f"⚠️ Gate 未通过，Top10 为空。{reason}\n\n")
    else:
        lines.append(_df_to_md_table(
            topN_df,
            cols=["rank", "ts_code", "name", "prob", "StrengthScore", "ThemeBoost", "board"],
        ))
        lines.append("\n")

    # 第2块：命中情况（prev_td -> trade_date）
    if prev_td:
        lines.append(f"## 《{prev_td} 预测：{trade_date} 命中情况》\n")
        prev_topN_df = _load_json_topN(outdir, prev_td)
        prev_hit_df, prev_metrics = _topN_to_hit_df(prev_topN_df, limit_df_current)
        if prev_topN_df is None or prev_topN_df.empty:
            lines.append("（未找到上一交易日预测文件或上一交易日 Top10 为空）\n\n")
        else:
            lines.append(_df_to_md_table(prev_hit_df, cols=["ts_code", "name", "prob", "命中", "板块"]))
            lines.append("\n")
    else:
        lines.append(f"## 《上一交易日预测：{trade_date} 命中情况》\n（找不到上一交易日）\n\n")

    # 第3块：强度列表（trade_date 所有涨停）
    lines.append(f"## 《{trade_date} 所有涨停股票的强度列表》\n")
    strength_limit_df = _join_limit_strength(limit_df_current, full_df)
    if strength_limit_df is None or strength_limit_df.empty:
        lines.append("(未能生成强度列表：limit_list 或 full_df 空)\n\n")
    else:
        lines.append(_df_to_md_table(
            strength_limit_df,
            cols=["ts_code", "name", "StrengthScore", "ThemeBoost", "board"],
        ))
        lines.append("\n")

    # 第4块：近10日命中率（predict_date -> next_trade_date 验证日）
    hist_df = _recent_hit_history(outdir, settings, ctx, max_days=10)
    if hist_df is not None and not hist_df.empty:
        lines.append("## 《近10日 Top10 命中率》\n")
        lines.append(_df_to_md_table(hist_df, cols=["日期", "命中数", "命中率", "当日涨停家数"]))
        lines.append("\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"✅ Outputs written: {md_path}")
