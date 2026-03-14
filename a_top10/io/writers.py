from __future__ import annotations

import json
import os
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
PROB_COL_CANDIDATES = ["prob_final", "prob", "Probability", "probability", "概率", "涨停概率"]
FINAL_SCORE_COL_CANDIDATES = ["final_score", "score", "Score", "最终得分"]
BOARD_COL_CANDIDATES = ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"]


# =========================
# P1: immutable + run meta
# =========================
def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_run_meta() -> Dict[str, str]:
    """
    统一从 GitHub Actions 环境变量取 run meta。
    - RUN_ID / GITHUB_RUN_ID
    - RUN_ATTEMPT / GITHUB_RUN_ATTEMPT
    - COMMIT_SHA / GITHUB_SHA
    """
    run_id = (os.getenv("RUN_ID") or os.getenv("GITHUB_RUN_ID") or "").strip()
    run_attempt = (os.getenv("RUN_ATTEMPT") or os.getenv("GITHUB_RUN_ATTEMPT") or "").strip()
    sha = (os.getenv("COMMIT_SHA") or os.getenv("GITHUB_SHA") or "").strip()

    if not run_id:
        # 兜底：时间戳（保证归档文件名不会冲突）
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


def _write_text_once(path: Path, text: str, *, force: bool = False, encoding: str = "utf-8") -> bool:
    """
    写一次锁定：
    - 若文件已存在且 force=False：不覆盖，返回 False
    - 否则写入，返回 True
    """
    _ensure_dir(path.parent)
    if path.exists() and not force:
        print(f"[SKIP] exists (write-once): {path}")
        return False
    path.write_text(text, encoding=encoding)
    print(f"[WRITE] {path}")
    return True


def _write_csv_once(df: pd.DataFrame, path: Path, *, force: bool = False) -> bool:
    _ensure_dir(path.parent)
    if path.exists() and not force:
        print(f"[SKIP] exists (write-once): {path}")
        return False
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path} rows={len(df)}")
    return True


def _append_csv_unique(
    df_new: pd.DataFrame,
    path: Path,
    *,
    key_cols: Sequence[str],
) -> Dict[str, int]:
    """
    append-only（不回写历史）：
    - 读取已有 CSV
    - 若 key_cols 相同的行已存在：跳过
    - 只追加新行
    返回统计：{"appended": x, "skipped": y, "total": z}
    """
    _ensure_dir(path.parent)

    if df_new is None or df_new.empty:
        return {"appended": 0, "skipped": 0, "total": int(path.exists())}

    df_new = df_new.copy()

    # 若文件不存在，直接写
    if not path.exists():
        df_new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[WRITE] {path} rows={len(df_new)} (new)")
        return {"appended": int(len(df_new)), "skipped": 0, "total": int(len(df_new))}

    # 读旧
    try:
        df_old = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except Exception:
        df_old = pd.DataFrame()

    # 确保 key 列存在
    for c in key_cols:
        if c not in df_old.columns:
            df_old[c] = ""
        if c not in df_new.columns:
            df_new[c] = ""

    def _key_df(d: pd.DataFrame) -> pd.Series:
        return d[list(key_cols)].astype(str).fillna("").agg("|".join, axis=1)

    old_keys = set(_key_df(df_old).tolist()) if (df_old is not None and not df_old.empty) else set()
    new_keys = _key_df(df_new).tolist()

    keep_mask = [k not in old_keys for k in new_keys]
    df_add = df_new.loc[keep_mask].copy()
    skipped = int(len(df_new) - len(df_add))

    if df_add.empty:
        print(f"[SKIP] append-only: no new rows for {path} (skipped={skipped})")
        return {"appended": 0, "skipped": skipped, "total": int(len(df_old))}

    # 追加写入（不重写旧内容）
    # 用 mode='a' 追加，避免覆盖
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        df_add.to_csv(f, index=False, header=False)

    total = int(len(df_old) + len(df_add))
    print(f"[APPEND] {path} appended={len(df_add)} skipped={skipped} total={total}")
    return {"appended": int(len(df_add)), "skipped": skipped, "total": total}


# =========================
# Markdown helpers
# =========================
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
        "prob_rule": "prob_rule",
        "prob_ml": "prob_ml",
        "prob_final": "prob_final",
        "final_score": "final_score",
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


def _ctx_trade_date(ctx: Any) -> str:
    if not isinstance(ctx, dict):
        return ""
    for k in ("trade_date", "TRADE_DATE", "asof", "date", "snapshot_date"):
        v = ctx.get(k)
        if v is None:
            continue
        s = _safe_str(v)
        if len(s) == 8 and s.isdigit():
            return s
    return ""


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


def _topN_to_hit_df(topN_df: Optional[pd.DataFrame], limit_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    if topN_df is None or topN_df.empty:
        metrics.update({"hit_count": 0, "top_count": 0, "limit_count": _count_limitups(limit_df), "hit_rate": ""})
        return pd.DataFrame(), metrics

    df = topN_df.copy()

    code_col = _first_existing_col(df, CODE_COL_CANDIDATES)
    if not code_col:
        metrics.update({"hit_count": 0, "top_count": len(df), "limit_count": _count_limitups(limit_df), "hit_rate": ""})
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
    limit_count = _count_limitups(limit_df)

    _, limit_ts, limit_c6 = _build_code_sets(limit_df, CODE_COL_CANDIDATES)

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


def _load_json_topN(outdir: Path, td: str) -> Optional[pd.DataFrame]:
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
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if "排名" in d.columns:
        pass
    elif "rank_limit" in d.columns:
        d["排名"] = d["rank_limit"]
    elif "rank" in d.columns:
        d["排名"] = d["rank"]
    else:
        d.insert(0, "排名", range(1, len(d) + 1))

    if "代码" not in d.columns:
        c = _first_existing_col(d, ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"])
        d["代码"] = d[c] if c else ""

    if "股票" not in d.columns:
        n = _first_existing_col(d, ["name", "stock_name", "名称", "证券名称", "股票简称"])
        d["股票"] = d[n] if n else ""

    if "prob_final" not in d.columns:
        p0 = _first_existing_col(d, ["prob_final", "Probability", "prob", "概率", "涨停概率"])
        d["prob_final"] = d[p0] if p0 else ""

    if "Probability" not in d.columns:
        p = _first_existing_col(d, ["Probability", "prob_final", "prob", "概率", "涨停概率"])
        d["Probability"] = d[p] if p else ""

    if "强度得分" not in d.columns:
        s = _first_existing_col(d, ["_strength", "StrengthScore", "强度得分", "强度"])
        d["强度得分"] = d[s] if s else ""

    if "题材加成" not in d.columns:
        t = _first_existing_col(d, ["ThemeBoost", "题材加成", "题材", "_theme"])
        d["题材加成"] = d[t] if t else ""

    if "板块" not in d.columns:
        b = _first_existing_col(d, ["board", "板块", "industry", "行业", "所属行业", "concept", "题材"])
        d["板块"] = d[b] if b else ""

    out_cols = ["排名", "代码", "股票", "prob_final", "Probability", "强度得分", "题材加成", "板块"]
    d = d[[c for c in out_cols if c in d.columns]].copy()

    if "强度得分" in d.columns:
        d["强度得分"] = pd.to_numeric(d["强度得分"], errors="coerce")
        d = d.sort_values(by="强度得分", ascending=False, na_position="last").reset_index(drop=True)
        d["排名"] = d.index + 1

    return d


def _join_limit_strength(limit_df: pd.DataFrame, full_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if limit_df is None or limit_df.empty:
        return None

    lcode = _first_existing_col(limit_df, CODE_COL_CANDIDATES)
    if not lcode:
        return None

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

    m = pd.merge(l[["ts_code"]], f, on="ts_code", how="left")

    if "name" not in m.columns:
        nc = _first_existing_col(m, NAME_COL_CANDIDATES)
        m["name"] = m[nc] if nc else ""

    if "board" not in m.columns:
        bc = _first_existing_col(m, BOARD_COL_CANDIDATES)
        m["board"] = m[bc] if bc else ""

    sort_cols = []
    if "StrengthScore" in m.columns:
        sort_cols = ["StrengthScore"]
    elif "final_score" in m.columns:
        sort_cols = ["final_score"]
    elif "score" in m.columns:
        sort_cols = ["score"]
    elif "prob_final" in m.columns:
        sort_cols = ["prob_final"]
    elif "prob" in m.columns:
        sort_cols = ["prob"]

    if sort_cols:
        m = m.sort_values(by=sort_cols, ascending=False)

    m = m.reset_index(drop=True)
    m.insert(0, "rank_limit", m.index + 1)
    return m


def _recent_hit_history(outdir: Path, settings, ctx, max_days: int = 10) -> pd.DataFrame:
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
        ldf_verify = _load_limit_df(settings, ctx, next_td)

        _, mres = _topN_to_hit_df(topN_df, ldf_verify)

        hit_count = mres.get("hit_count")
        top_count = mres.get("top_count")
        hit_rate = mres.get("hit_rate")

        if hit_count is None or top_count is None:
            continue

        rows.append({
            "日期": d,
            "命中数": int(hit_count),
            "命中率": str(hit_rate) if hit_rate is not None else "",
            "当日涨停家数": _count_limitups(ldf_verify),
        })
        if len(rows) >= max_days:
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _standardize_topN_for_csv(topN_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    统一 topN / full ranking 字段，兼容 V1/V2：
    rank, ts_code, name, prob_rule, prob_ml, prob_final, prob, Probability,
    final_score, score, StrengthScore, ThemeBoost, board
    其中：
    - prob_final 是 V2 主概率口径
    - Probability / prob 是兼容别名
    - final_score 是 V2 主终排口径
    """
    base_cols = [
        "rank", "ts_code", "name",
        "prob_rule", "prob_ml", "prob_final", "prob", "Probability",
        "final_score", "score",
        "StrengthScore", "ThemeBoost", "board",
    ]
    if topN_df is None or topN_df.empty:
        return pd.DataFrame(columns=base_cols)

    df = topN_df.copy()

    # rank
    if "rank" not in df.columns:
        if "排名" in df.columns:
            df["rank"] = df["排名"]
        else:
            df["rank"] = range(1, len(df) + 1)

    # ts_code
    c = _first_existing_col(df, CODE_COL_CANDIDATES)
    if c and c != "ts_code":
        df["ts_code"] = df[c]
    elif "ts_code" not in df.columns:
        df["ts_code"] = ""

    # name
    n = _first_existing_col(df, NAME_COL_CANDIDATES)
    if n and n != "name":
        df["name"] = df[n]
    elif "name" not in df.columns:
        df["name"] = ""

    # V2 probability fields
    p_final = _first_existing_col(df, ["prob_final", "Probability", "prob", "probability", "概率", "涨停概率"])
    if p_final and p_final != "prob_final":
        df["prob_final"] = df[p_final]
    elif "prob_final" not in df.columns:
        df["prob_final"] = ""

    if "Probability" not in df.columns:
        df["Probability"] = df["prob_final"]
    if "prob" not in df.columns:
        df["prob"] = df["prob_final"]

    if "prob_rule" not in df.columns:
        df["prob_rule"] = ""
    if "prob_ml" not in df.columns:
        df["prob_ml"] = ""

    # V2 final score fields
    s_final = _first_existing_col(df, ["final_score", "score", "Score", "最终得分"])
    if s_final and s_final != "final_score":
        df["final_score"] = df[s_final]
    elif "final_score" not in df.columns:
        df["final_score"] = ""

    if "score" not in df.columns:
        df["score"] = df["final_score"]

    # board
    b = _first_existing_col(df, BOARD_COL_CANDIDATES)
    if b and b != "board":
        df["board"] = df[b]
    elif "board" not in df.columns:
        df["board"] = ""

    # StrengthScore / ThemeBoost
    if "StrengthScore" not in df.columns:
        if "强度得分" in df.columns:
            df["StrengthScore"] = df["强度得分"]
        else:
            df["StrengthScore"] = ""
    if "ThemeBoost" not in df.columns:
        if "题材加成" in df.columns:
            df["ThemeBoost"] = df["题材加成"]
        else:
            df["ThemeBoost"] = ""

    # normalize numeric-ish fields, but keep empty string if full column missing
    for col in ["prob_rule", "prob_ml", "prob_final", "prob", "Probability", "final_score", "score", "StrengthScore", "ThemeBoost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    use_cols = [c for c in base_cols if c in df.columns]
    df = df[use_cols].copy()
    return df


def _merge_strengthscore_from_step3(full_std: Optional[pd.DataFrame], outdir: Path, trade_date: str) -> Optional[pd.DataFrame]:
    """
    兜底修复：
    - 如果 full_std 的 StrengthScore 缺失/全 0，则尝试从 step3 输出 CSV 回填。
    - 只回填 StrengthScore 为 NaN/空/0 的位置（避免覆盖上游正确值）。
    - 主要目标：修复 outputs/decisio/pred_decisio_latest.csv StrengthScore 全 0。
    """
    if full_std is None or full_std.empty:
        return full_std

    if "ts_code" not in full_std.columns:
        return full_std

    # 如果 StrengthScore 列都不存在，先补出来（后续回填）
    if "StrengthScore" not in full_std.columns:
        full_std = full_std.copy()
        full_std["StrengthScore"] = ""

    base_num = pd.to_numeric(full_std["StrengthScore"], errors="coerce")
    nonzero_cnt = int((base_num.fillna(0) != 0).sum())
    if nonzero_cnt > 0:
        # 已经有非零值，仍允许部分回填（只对 0/NaN 的行）
        pass

    # step3 候选文件（按优先级）
    candidates: List[Path] = [
        outdir / f"step3_strength_{trade_date}.csv",
        outdir / "step3_strength.csv",
        outdir / f"outputs/step3_strength_{trade_date}.csv",
        outdir / "outputs/step3_strength.csv",
    ]
    step3_path = None
    for p in candidates:
        try:
            if p.exists():
                step3_path = p
                break
        except Exception:
            continue

    if step3_path is None:
        return full_std

    df_step3 = _read_csv_guess(step3_path)
    if df_step3 is None or df_step3.empty:
        return full_std

    code_col = _first_existing_col(df_step3, CODE_COL_CANDIDATES)
    if not code_col:
        return full_std

    strength_col = _first_existing_col(df_step3, ["StrengthScore", "强度得分", "_strength", "strength", "强度", "strengthscore"])
    if not strength_col:
        return full_std

    s = df_step3[[code_col, strength_col]].copy()
    if code_col != "ts_code":
        s["ts_code"] = s[code_col]
    s["StrengthScore_step3"] = pd.to_numeric(s[strength_col], errors="coerce")
    s = s[["ts_code", "StrengthScore_step3"]].dropna(subset=["ts_code"])
    s = s.drop_duplicates(subset=["ts_code"], keep="last")

    m = full_std.merge(s, on="ts_code", how="left")

    base = pd.to_numeric(m["StrengthScore"], errors="coerce")
    fill_mask = base.isna() | (base == 0)

    before_fill = int(fill_mask.sum())
    if before_fill <= 0:
        return full_std

    # 只有 step3 有值的才回填
    step3_v = m["StrengthScore_step3"]
    can_fill = fill_mask & step3_v.notna()
    filled = int(can_fill.sum())

    if filled > 0:
        m.loc[can_fill, "StrengthScore"] = m.loc[can_fill, "StrengthScore_step3"]
        m["StrengthScore"] = pd.to_numeric(m["StrengthScore"], errors="coerce")

    m = m.drop(columns=["StrengthScore_step3"], errors="ignore")

    print(f"[FIX] StrengthScore backfill from {step3_path.name}: fill_candidates={before_fill} filled={filled}")
    return m


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn) -> None:
    outdir = getattr(getattr(settings, "io", None), "outputs_dir", None)
    if not outdir:
        outdir = "outputs"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # P1 meta
    run_meta = _get_run_meta()
    force_overwrite = str(os.getenv("FORCE_OVERWRITE", "0")).strip() in ("1", "true", "True", "YES", "yes")

    # learning + warehouse dirs
    learning_dir = outdir / "learning"
    wh_pred_dir = outdir / "_warehouse" / "pred_top10"
    _ensure_dir(learning_dir)
    _ensure_dir(wh_pred_dir)

    topN_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None
    limit_up_table_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        topN_df = _pick_first_not_none(topn, ["topN", "topn", "TopN", "top"])
        full_df = topn.get("full") if "full" in topn else None
        limit_up_table_df = topn.get("limit_up_table") if "limit_up_table" in topn else None
    else:
        topN_df = topn

    topN_df = _to_df(topN_df)
    full_df = _to_df(full_df)
    limit_up_table_df = _to_df(limit_up_table_df)

    calendar = _list_trade_dates(settings)
    prev_td, next_td = _prev_next_trade_date_with_fallback(calendar, trade_date)

    limit_df_current = _load_limit_df(settings, ctx, trade_date)

    _hit_df_same_day, metrics_same_day = _topN_to_hit_df(topN_df, limit_df_current)

    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "verify_date": next_td,
        "gate": gate,
        "topN": [] if topN_df is None else topN_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "limit_up_table": [] if limit_up_table_df is None else limit_up_table_df.to_dict(orient="records"),
        "learn": learn,
        "metrics": metrics_same_day,
        "metrics_same_day": metrics_same_day,
        "run_meta": run_meta,
    }

    # ================
    # P1: write-once daily json/md (freeze history)
    # ================
    json_path = outdir / f"predict_top10_{trade_date}.json"
    _write_text_once(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        force=force_overwrite,
        encoding="utf-8",
    )

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
            cols=["rank", "ts_code", "name", "prob_final", "final_score", "StrengthScore", "ThemeBoost", "board"],
        ))
        lines.append("\n")

    # 第2块：强度列表（trade_date 所有涨停）
    lines.append(f"## 《{trade_date} 所有涨停股票的强度列表》\n")

    strength_limit_df = None
    if limit_up_table_df is not None and not limit_up_table_df.empty:
        strength_limit_df = _standardize_strength_table(limit_up_table_df)
    else:
        j = _join_limit_strength(limit_df_current, full_df)
        if j is not None and not j.empty:
            strength_limit_df = _standardize_strength_table(j)

    if strength_limit_df is None or strength_limit_df.empty:
        lines.append("(未能生成强度列表：limit_list 或 full_df 空，且未提供 limit_up_table)\n\n")
    else:
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
        lines.append(_df_to_md_table(prev_hit_df, cols=["ts_code", "name", "prob_final", "命中", "板块"]))
        lines.append("\n")

    # 第4块：近10日命中率（predict_date -> next_trade_date 验证日）
    hist_df = _recent_hit_history(outdir, settings, ctx, max_days=10)
    if hist_df is not None and not hist_df.empty:
        lines.append("## 《近10日 Top10 命中率》\n")
        lines.append(_df_to_md_table(hist_df, cols=["日期", "命中数", "命中率", "当日涨停家数"]))
        lines.append("\n")

    md_text = "\n".join(lines)
    _write_text_once(md_path, md_text, force=force_overwrite, encoding="utf-8")

    # latest.md（允许覆盖，属于“当前展示”）
    latest = outdir / "latest.md"
    latest.write_text(md_text, encoding="utf-8")
    print(f"[WRITE] {latest} (latest)")

    # ================
    # P1: CSV outputs
    # - immutable warehouse
    # - learning per-day (write-once)
    # - history append-only
    # ================
    topN_std = _standardize_topN_for_csv(topN_df)

    # enrich
    if topN_std is not None and not topN_std.empty:
        topN_std = topN_std.copy()
        topN_std.insert(0, "trade_date", trade_date)
        topN_std.insert(1, "verify_date", next_td)
        topN_std["run_id"] = run_meta["run_id"]
        topN_std["run_attempt"] = run_meta["run_attempt"]
        topN_std["commit_sha"] = run_meta["commit_sha"]
        topN_std["generated_at_utc"] = run_meta["generated_at_utc"]

    # =========================
    # ✅ NEW: decisio full ranking outputs (TopK/Full list)
    # outputs/decisio/
    # - pred_decisio_{trade_date}.csv   (write-once, archive)
    # - pred_decisio_latest.csv         (overwrite, latest)
    # =========================
    full_std = _standardize_topN_for_csv(full_df)
    full_std = _merge_strengthscore_from_step3(full_std, outdir, trade_date)

    # enrich（与 topN_std 保持一致字段，保证下游可直接复用）
    if full_std is not None and not full_std.empty:
        full_std = full_std.copy()
        full_std.insert(0, "trade_date", trade_date)
        full_std.insert(1, "verify_date", next_td)
        full_std["run_id"] = run_meta["run_id"]
        full_std["run_attempt"] = run_meta["run_attempt"]
        full_std["commit_sha"] = run_meta["commit_sha"]
        full_std["generated_at_utc"] = run_meta["generated_at_utc"]

        decisio_dir = outdir / "decisio"
        _ensure_dir(decisio_dir)

        # 1) 日归档（write-once；除非 FORCE_OVERWRITE=1 才允许覆盖）
        decisio_dated = decisio_dir / f"pred_decisio_{trade_date}.csv"
        _write_csv_once(full_std, decisio_dated, force=False)

        # 2) latest（覆盖，用于下游拉取）
        decisio_latest = decisio_dir / "pred_decisio_latest.csv"
        full_std.to_csv(decisio_latest, index=False, encoding="utf-8-sig")
        print(f"[WRITE] {decisio_latest} rows={len(full_std)} (latest)")
    else:
        print("[WARN] full_df empty -> skip outputs/decisio")

    # 1) immutable archive (always write, never overwrite)
    wh_csv = wh_pred_dir / f"pred_top10_{trade_date}_{run_meta['run_id']}.csv"
    _write_csv_once(topN_std, wh_csv, force=False)

    # 2) learning per-day (STRICT write-once, NEVER overwrite)
    learn_csv = learning_dir / f"pred_top10_{trade_date}.csv"
    _write_csv_once(topN_std, learn_csv, force=False)

    # 2b) learning latest (overwrite OK, for downstream consumption)
    learn_latest = learning_dir / "pred_top10_latest.csv"
    _ensure_dir(learn_latest.parent)
    (topN_std if topN_std is not None else pd.DataFrame()).to_csv(learn_latest, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {learn_latest} rows={0 if topN_std is None else len(topN_std)} (latest)")

    # 3) pred_top10_history.csv append-only (no rewrite)
    hist_path = learning_dir / "pred_top10_history.csv"
    # 只按 (trade_date, ts_code) 去重：一旦该票当日已记录，就不再追加（防止历史变动）
    if topN_std is not None and not topN_std.empty:
        hist_stats = _append_csv_unique(
            topN_std,
            hist_path,
            key_cols=["trade_date", "ts_code"],
        )
        print(f"[HISTORY] pred_top10_history.csv appended={hist_stats['appended']} skipped={hist_stats['skipped']} total={hist_stats['total']}")

    # 4) _last_run.txt (overwrite is OK)
    last_run = learning_dir / "_last_run.txt"
    last_text = (
        f"trade_date={trade_date}\n"
        f"verify_date={next_td}\n"
        f"run_id={run_meta['run_id']}\n"
        f"run_attempt={run_meta['run_attempt']}\n"
        f"commit_sha={run_meta['commit_sha']}\n"
        f"generated_at_utc={run_meta['generated_at_utc']}\n"
    )
    last_run.write_text(last_text, encoding="utf-8")
    print(f"[WRITE] {last_run} (latest)")

    print(f"✅ Outputs written: {md_path}")
