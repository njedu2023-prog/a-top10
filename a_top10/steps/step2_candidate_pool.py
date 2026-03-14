#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from a_top10.config import Settings


# ============================================================
# V2 constants
# ============================================================
CODE_COL_CANDIDATES = ["ts_code", "TS_CODE", "code", "CODE", "证券代码", "股票代码", "代码"]
NAME_COL_CANDIDATES = ["name", "名称", "股票简称", "ts_name", "证券名称"]
INDUSTRY_COL_CANDIDATES = ["industry", "行业", "industry_name", "所属行业"]


# ============================================================
# basic utils
# ============================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_default(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "<unserializable>"


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _resolve_outputs_dir(s: Settings) -> Path:
    try:
        io_cfg = getattr(s, "io", None)
        if io_cfg is not None and getattr(io_cfg, "outputs_dir", None):
            return Path(getattr(io_cfg, "outputs_dir"))
    except Exception:
        pass
    return Path("outputs")


def _pick_trade_date(s: Settings, ctx: Dict[str, Any]) -> str:
    td = str(ctx.get("trade_date", "") or "").strip()
    if td:
        return td
    try:
        v = getattr(s, "trade_date", "")
        if v:
            return str(v).strip()
    except Exception:
        pass
    return "unknown"


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _safe_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if df is None or df.empty or not col or col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna("").map(lambda x: str(x).strip())


def _normalize_ts_code_value(x: Any) -> str:
    s = str(x or "").strip().upper()
    if not s:
        return ""

    if "." in s:
        left, right = s.split(".", 1)
        left = left.strip()
        right = right.strip().replace("SSE", "SH").replace("SZSE", "SZ")
        if len(left) == 6 and left.isdigit() and right in ("SH", "SZ", "BJ"):
            return f"{left}.{right}"
        return s

    if s.startswith(("SZ", "SH", "BJ")) and len(s) >= 8:
        exch = s[:2]
        code = s[2:8]
        if code.isdigit():
            return f"{code}.{exch}"

    if len(s) >= 8 and s[:6].isdigit() and s[6:8] in ("SZ", "SH", "BJ"):
        return f"{s[:6]}.{s[6:8]}"

    if len(s) == 6 and s.isdigit():
        return s

    return s


def _normalize_code_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code"])

    out = df.copy()
    code_col = _first_existing_col(out, CODE_COL_CANDIDATES)
    if code_col is None:
        raise RuntimeError("V2 contract violated: candidate source missing code column")

    out["ts_code"] = _safe_series(out, code_col).map(_normalize_ts_code_value)
    out = out[out["ts_code"] != ""].copy()
    return out


def _ensure_name_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name" in out.columns:
        out["name"] = _safe_series(out, "name")
        return out

    name_col = _first_existing_col(out, NAME_COL_CANDIDATES)
    out["name"] = _safe_series(out, name_col)
    return out


def _ensure_industry_board_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ind_col = _first_existing_col(out, INDUSTRY_COL_CANDIDATES)
    out["industry"] = _safe_series(out, ind_col) if ind_col else pd.Series([""] * len(out), index=out.index, dtype="string")

    if "board" in out.columns:
        board_s = _safe_series(out, "board")
        out["board"] = board_s.where(board_s != "", out["industry"])
    else:
        out["board"] = out["industry"]

    return out


def _filter_bad_names(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    stats = {"drop_st": 0, "drop_delist": 0}

    if "name" not in out.columns:
        return out, stats

    name_s = _safe_series(out, "name")

    mask_st = name_s.str.contains("ST", case=False, regex=False)
    stats["drop_st"] = int(mask_st.sum())
    out = out.loc[~mask_st].copy()

    name_s2 = _safe_series(out, "name")
    mask_delist = name_s2.str.contains("退", regex=False) | name_s2.str.contains("退市", regex=False)
    stats["drop_delist"] = int(mask_delist.sum())
    out = out.loc[~mask_delist].copy()

    return out, stats


def _merge_stock_basic(base_df: pd.DataFrame, stock_basic: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    dbg: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "stock_basic_rows": 0,
        "industry_nonblank_ratio_after": 0.0,
    }

    if stock_basic is None or stock_basic.empty:
        dbg["reason"] = "stock_basic missing"
        return base_df, dbg

    sb = _normalize_code_column(stock_basic.copy())
    dbg["stock_basic_rows"] = int(len(sb))

    name_col = _first_existing_col(sb, NAME_COL_CANDIDATES)
    ind_col = _first_existing_col(sb, INDUSTRY_COL_CANDIDATES)

    keep_cols = ["ts_code"]
    if name_col:
        keep_cols.append(name_col)
    if ind_col:
        keep_cols.append(ind_col)

    sb = sb[keep_cols].copy()
    rename_map: Dict[str, str] = {}
    if name_col and name_col != "name":
        rename_map[name_col] = "name_sb"
    elif name_col == "name":
        rename_map[name_col] = "name_sb"

    if ind_col and ind_col != "industry":
        rename_map[ind_col] = "industry_sb"
    elif ind_col == "industry":
        rename_map[ind_col] = "industry_sb"

    sb = sb.rename(columns=rename_map)

    out = base_df.merge(sb, on="ts_code", how="left")

    if "name" not in out.columns:
        out["name"] = ""
    if "name_sb" in out.columns:
        base_name = _safe_series(out, "name")
        sb_name = _safe_series(out, "name_sb")
        out["name"] = base_name.where(base_name != "", sb_name)
        out = out.drop(columns=["name_sb"], errors="ignore")

    if "industry" not in out.columns:
        out["industry"] = ""
    if "industry_sb" in out.columns:
        base_ind = _safe_series(out, "industry")
        sb_ind = _safe_series(out, "industry_sb")
        out["industry"] = base_ind.where(base_ind != "", sb_ind)
        out = out.drop(columns=["industry_sb"], errors="ignore")

    ratio = 0.0
    if len(out) > 0 and "industry" in out.columns:
        ratio = float((_safe_series(out, "industry") != "").mean())

    dbg["ok"] = True
    dbg["industry_nonblank_ratio_after"] = ratio
    return out, dbg


def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = _ensure_name_col(out)
    out = _ensure_industry_board_cols(out)

    front = [c for c in ["ts_code", "name", "industry", "board"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest].copy()

    out["ts_code"] = _safe_series(out, "ts_code").map(_normalize_ts_code_value)
    out = out[out["ts_code"] != ""].copy()
    out = out.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)

    return out


def _get_candidate_source(ctx: Dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """
    V2 只接受最小主链输入：
    1) ctx["limit_list_d"]
    2) ctx["limit_df"]
    不再扫大量旧链路键名。
    """
    for key in ["limit_list_d", "limit_df"]:
        v = ctx.get(key)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v.copy(), key
    raise RuntimeError("V2 contract violated: step2 requires ctx['limit_list_d'] or ctx['limit_df'] as DataFrame")


# ============================================================
# Step2 main
# ============================================================
def step2_build_candidates(s: Settings, ctx: Dict[str, Any]) -> pd.DataFrame:
    if ctx is None or not isinstance(ctx, dict):
        raise RuntimeError("V2 contract violated: step2_build_candidates requires dict ctx")

    trade_date = _pick_trade_date(s, ctx)
    out_dir = _resolve_outputs_dir(s)
    _ensure_dir(out_dir)

    debug: Dict[str, Any] = {
        "trade_date": trade_date,
        "base_source": "",
        "base_rows": 0,
        "stock_basic_used": False,
        "filters": {"drop_st": 0, "drop_delist": 0},
        "industry_merge": {},
        "final_rows": 0,
        "final_cols": [],
        "debug_file": str(out_dir / f"debug_step2_candidate_{trade_date}.json"),
        "out_csv": str(out_dir / f"step2_candidates_{trade_date}.csv"),
    }

    base_raw, base_source = _get_candidate_source(ctx)
    debug["base_source"] = base_source
    debug["base_rows"] = int(len(base_raw))

    base_df = _normalize_code_column(base_raw)
    base_df = _ensure_name_col(base_df)
    base_df = _ensure_industry_board_cols(base_df)

    stock_basic = ctx.get("stock_basic")
    if isinstance(stock_basic, pd.DataFrame) and not stock_basic.empty:
        debug["stock_basic_used"] = True
        base_df, merge_dbg = _merge_stock_basic(base_df, stock_basic)
        debug["industry_merge"] = merge_dbg
    else:
        debug["industry_merge"] = {
            "ok": False,
            "reason": "stock_basic missing",
            "stock_basic_rows": 0,
            "industry_nonblank_ratio_after": float((_safe_series(base_df, "industry") != "").mean()) if len(base_df) else 0.0,
        }

    candidates_df = _finalize_schema(base_df)
    candidates_df, filter_stats = _filter_bad_names(candidates_df)
    debug["filters"] = filter_stats

    candidates_df = _finalize_schema(candidates_df)

    debug["final_rows"] = int(len(candidates_df))
    debug["final_cols"] = list(candidates_df.columns)

    candidates_df.to_csv(Path(debug["out_csv"]), index=False, encoding="utf-8-sig")
    _write_json(Path(debug["debug_file"]), debug)

    ctx["candidates"] = candidates_df
    ctx["step2"] = candidates_df
    ctx["candidate_pool"] = candidates_df
    ctx["step2_candidates"] = candidates_df

    return candidates_df


def run(df: Any, s: Optional[Settings] = None, ctx: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    保留旧函数名外壳，但内部只服务 V2。
    """
    if ctx is None:
        ctx = {}
    if isinstance(df, pd.DataFrame):
        ctx["limit_list_d"] = df

    if s is None:
        s = Settings()

    return step2_build_candidates(s, ctx)


if __name__ == "__main__":
    print("Step2 V2 module loaded successfully.")
