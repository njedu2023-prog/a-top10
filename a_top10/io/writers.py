from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


CODE_COL_CANDIDATES = [
    "ts_code",
    "code",
    "TS_CODE",
    "证券代码",
    "股票代码",
]
NAME_COL_CANDIDATES = ["name", "stock_name", "名称", "股票", "证券名称", "股票简称"]
PROB_COL_CANDIDATES = ["prob", "Probability", "概率", "涨停概率"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"]


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
        "rank_limit": "排名",
        "排名": "排名",

        "ts_code": "代码",
        "code": "代码",
        "代码": "代码",

        "name": "股票",
        "stock_name": "股票",
        "股票": "股票",

        "score": "Score",
        "Score": "Score",

        "prob": "Probability",
        "Probability": "Probability",

        "StrengthScore": "强度得分",
        "强度得分": "强度得分",

        "ThemeBoost": "题材加成",
        "题材加成": "题材加成",

        "board": "板块",
        "板块": "板块",
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
    """只按已存在的交易日历计算上下交易日"""
    if calendar and trade_date in calendar:
        i = calendar.index(trade_date)
        prev_td = calendar[i - 1] if i - 1 >= 0 else ""
        next_td = calendar[i + 1] if i + 1 < len(calendar) else ""
        return prev_td, next_td
    return "", ""


def _prev_next_trade_date_with_fallback(calendar: List[str], trade_date: str) -> Tuple[str, str]:
    """用于报告显示：next_td 可以用工作日规则兜底出来"""
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


def _load_json_topN(outdir: Path, td: str) -> Optional[pd.DataFrame]:
    # ✅ 修复：td 为空时不要读 predict_top10_.json
    if not td:
        return None
    p = outdir / f"predict_top10_{td}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    top = payload.get("topN") or payload.get("topn") or []
    return _to_df(top)


def _standardize_strength_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    把“强度列表”标准化成报告需要的列：
      排名 / 代码 / 股票 / Probability / 强度得分 / 题材加成 / 板块
    兼容 Step6 新增 limit_up_table（中文列）以及旧 join 输出（英文列）。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # rank
    if "排名" in d.columns:
        pass
    elif "rank_limit" in d.columns:
        d["排名"] = d["rank_limit"]
    elif "rank" in d.columns:
        d["排名"] = d["rank"]
    else:
        d.insert(0, "排名", range(1, len(d) + 1))

    # code
    if "代码" not in d.columns:
        c = _first_existing_col(d, ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"])
        if c:
            d["代码"] = d[c]
        else:
            d["代码"] = ""

    # name
    if "股票" not in d.columns:
        n = _first_existing_col(d, ["name", "stock_name", "名称", "证券名称", "股票简称"])
        if n:
            d["股票"] = d[n]
        else:
            d["股票"] = ""

    # prob
    if "Probability" not in d.columns:
        p = _first_existing_col(d, ["prob", "Probability", "概率", "涨停概率"])
        if p:
            d["Probability"] = d[p]
        else:
            d["Probability"] = ""

    # strength
    if "强度得分" not in d.columns:
    # 优先使用 Step6 真实强度字段
        s = _first_existing_col(d, ["_strength", "StrengthScore", "强度得分", "强度"])
        if s:
            d["强度得分"] = d[s]
        else:
            d["强度得分"] = ""

    # theme
    if "题材加成" not in d.columns:
        t = _first_existing_col(d, ["ThemeBoost", "题材加成", "题材", "_theme"])
        if t:
            d["题材加成"] = d[t]
        else:
            d["题材加成"] = ""

    # board
    if "板块" not in d.columns:
        b = _first_existing_col(d, ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"])
        if b:
            d["板块"] = d[b]
        else:
            d["板块"] = ""

    # 输出列顺序固定
    out_cols = ["排名", "代码", "股票", "Probability", "强度得分", "题材加成", "板块"]
    d = d[[c for c in out_cols if c in d.columns]].copy()

    # === 按强度得分降序排序，并重排排名 ===
    if "强度得分" in d.columns:
        d["强度得分"] = pd.to_numeric(d["强度得分"], errors="coerce")
        d = d.sort_values(by="强度得分", ascending=False, na_position="last").reset_index(drop=True)
        d["排名"] = d.index + 1

    return d


def _join_limit_strength(limit_df: pd.DataFrame, full_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    旧逻辑 fallback：
    - 筛出当日涨停股
    - merge full_df 带回 StrengthScore / ThemeBoost / board / prob（如果有）
    - 并按 StrengthScore(优先) / score / prob 排序
    """
    if limit_df is None or limit_df.empty:
        return None

    lcode = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not lcode:
        return None

    # full_df 不存在：至少输出代码（其余空）
    if full_df is None or full_df.empty:
        df = limit_df[[lcode]].copy()
        df = df.rename(columns={lcode: "ts_code"})
        df["name"] = ""
        df["prob"] = ""
        df["StrengthScore"] = ""
        df["ThemeBoost"] = ""
        df["board"] = ""
        return df

    fcode = _first_existing_col(full_df, CODE_COL_CANDIDATES)
    if not fcode:
        return None

    l = limit_df[[lcode]].copy()
    f = full_df.copy()

    if lcode != "ts_code":
        l["ts_code"] = l[lcode]
    if fcode != "ts_code":
        f["ts_code"] = f[fcode]

    # merge
    m = pd.merge(l[["ts_code"]], f, on="ts_code", how="left")

    # name fallback
    if "name" not in m.columns:
        nc = _first_existing_col(m, NAME_COL_CANDIDATES)
        if nc:
            m["name"] = m[nc]
        else:
            m["name"] = ""

    # board fallback
    if "board" not in m.columns:
        bc = _first_existing_col(m, BOARD_COL_CANDIDATES)
        if bc:
            m["board"] = m[bc]
        else:
            m["board"] = ""

    # sort priority
    sort_cols = []
    if "StrengthScore" in m.columns:
        sort_cols = ["StrengthScore"]
    elif "score" in m.columns:
        sort_cols = ["score"]
    elif "prob" in m.columns:
        sort_cols = ["prob"]

    if sort_cols:
        m = m.sort_values(by=sort_cols, ascending=False)

    # rank for display
    m = m.reset_index(drop=True)
    m.insert(0, "rank_limit", m.index + 1)
    return m



def _recent_hit_history(outdir: Path, settings, ctx, max_days: int = 10) -> pd.DataFrame:
    """
    近10日命中率口径：predict_date -> next_trade_date 的 limit_list。

    注意：latest（未来没有 next_td 的）会被跳过，避免用空 limit_list 得出误导 0。
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
            continue

        topN_df = _load_json_topN(outdir, d)
        ldf = _load_limit_df(settings, ctx, next_td)
        _, mres = _topN_to_hit_df(topN_df, ldf)

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
    limit_up_table_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        topN_df = _pick_first_not_none(topn, ["topN", "topn", "TopN", "top"])
        full_df = topn.get("full") if "full" in topn else None
        # ✅ Step6 新增：limit_up_table（优先用于第3块）
        limit_up_table_df = topn.get("limit_up_table") if "limit_up_table" in topn else None
    else:
        topN_df = topn

    topN_df = _to_df(topN_df)
    full_df = _to_df(full_df)
    limit_up_table_df = _to_df(limit_up_table_df)

    # trade calendar
    calendar = _list_trade_dates(settings)
    prev_td, next_td = _prev_next_trade_date_with_fallback(calendar, trade_date)

    # 本日 limit_list（用于第2块验证命中；第3块强度列表 fallback）
    limit_df_current = _load_limit_df(settings, ctx, trade_date)

    # JSON payload：保持兼容，不大动 payload 结构（metrics 仍按“预测日验证预测日”的旧口径）
    _hit_df_same_day, metrics_same_day = _topN_to_hit_df(topN_df, limit_df_current)

    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "verify_date": next_td,  # 下一交易日：预测目标日（report 第1块标题里的那个）
        "gate": gate,
        "topN": [] if topN_df is None else topN_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        # ✅ 可选附带：limit_up_table（不影响旧解析）
        "limit_up_table": [] if limit_up_table_df is None else limit_up_table_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics_same_day,
        "metrics_same_day": metrics_same_day,
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    # Markdown（四块中文标题）
    md_path = outdir / f"predict_top10_{trade_date}.md"
    lines = [f"# {trade_date} 预测报告\n"]

    # 第1块：预测（trade_date -> next_td）
    lines.append(f"## 《{trade_date} 预测：{next_td} 涨停 TOP 10》\n")

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


    # 第2块：强度列表（trade_date 所有涨停）
    lines.append(f"## 《{trade_date} 所有涨停股票的强度列表》\n")

    # ✅ 优先用 Step6 输出 limit_up_table（字段已经对齐报告）
    strength_limit_df = None
    if limit_up_table_df is not None and not limit_up_table_df.empty:
        strength_limit_df = _standardize_strength_table(limit_up_table_df)
    else:
        # fallback：旧 join 逻辑
        j = _join_limit_strength(limit_df_current, full_df)
        if j is not None and not j.empty:
            strength_limit_df = _standardize_strength_table(j)

    if strength_limit_df is None or strength_limit_df.empty:
        lines.append("(未能生成强度列表：limit_list 或 full_df 空，且未提供 limit_up_table)\n\n")
    else:
        # 输出统一列：排名/代码/股票/Probability/强度得分/题材加成/板块
        lines.append(_df_to_md_table(
            strength_limit_df,
            cols=["排名", "代码", "股票", "Probability", "强度得分", "题材加成", "板块"],
        ))
        lines.append("\n")

    # 第3块：命中情况（prev_td -> trade_date）
    prev_title = prev_td if prev_td else "上一交易日"
    lines.append(f"## 《{prev_title} 预测：{trade_date} 命中情况》\n")

    prev_topN_df = _load_json_topN(outdir, prev_td)
    prev_hit_df, _prev_metrics = _topN_to_hit_df(prev_topN_df, limit_df_current)
    if prev_topN_df is None or prev_topN_df.empty:
        lines.append("（未找到上一交易日预测文件或上一交易日 Top10 为空）\n\n")
    else:
        lines.append(_df_to_md_table(prev_hit_df, cols=["ts_code", "name", "prob", "命中", "板块"]))
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
