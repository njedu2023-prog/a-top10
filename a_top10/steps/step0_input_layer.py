from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


# ============================================================
# Step0: Input Layer (Clean & Robust)
# - Load snapshot CSV tables from data repo warehouse/snapshot dir
# - Normalize ts_code + industry columns
# - Build "universe" base table with guaranteed industry fields
# - Provide rich debug info for downstream diagnosis
# ============================================================


# -------------------------
# IO utils
# -------------------------
def _read_csv_if_exists(p: Path) -> pd.DataFrame:
    """Read CSV with multiple encoding fallbacks. Return empty DF if missing/unreadable."""
    if not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc)
        except Exception:
            pass
    try:
        # last resort: let pandas guess
        return pd.read_csv(p, dtype=str)
    except Exception:
        return pd.DataFrame()


def _dir_has_any_csv(d: Path, filenames: List[str]) -> bool:
    if not d.exists() or (not d.is_dir()):
        return False
    for fn in filenames:
        if (d / fn).exists():
            return True
    return False


def _dir_has_all_csv(d: Path, filenames: List[str]) -> bool:
    if not d.exists() or (not d.is_dir()):
        return False
    for fn in filenames:
        if not (d / fn).exists():
            return False
    return True


def _pick_snapshot_dir(s: Settings, trade_date: str) -> Tuple[Path, List[str]]:
    """
    Choose snapshot dir robustly:
    - primary: s.data_repo.snapshot_dir(trade_date)
    - fallbacks: common warehouse/snap shapes
    Rule: MUST hit at least one key CSV to avoid selecting empty directory.
    """
    tried: List[str] = []

    key_files = [
        "daily.csv",
        "daily_basic.csv",
        "limit_list_d.csv",
        "limit_break_d.csv",
        "hot_boards.csv",
        "top_list.csv",
        "stock_basic.csv",
    ]

    primary = s.data_repo.snapshot_dir(trade_date)
    tried.append(str(primary))
    if primary.exists() and primary.is_dir() and _dir_has_any_csv(primary, key_files):
        return primary, tried

    year = trade_date[:4]

    # repo_name if available
    repo_name = None
    try:
        repo_name = getattr(getattr(s, "data_repo", None), "name", None)
    except Exception:
        repo_name = None
    if not repo_name:
        repo_name = "a-share-top3-data"

    candidates: List[Path] = [
        Path("snap") / trade_date,
        Path("snapshots") / trade_date,
        Path("data_repo") / "snapshots" / trade_date,
        Path("_warehouse") / repo_name / "data" / "raw" / year / trade_date,
        Path("_warehouse") / repo_name / "raw" / year / trade_date,
        Path("_warehouse") / "data" / "raw" / year / trade_date,
        Path("snap"),
        Path("snapshots"),
    ]

    # Prefer directory that has MORE of the key files.
    best: Optional[Path] = None
    best_score = -1

    for d in candidates:
        tried.append(str(d))
        if not d.exists() or not d.is_dir():
            continue
        score = sum(int((d / fn).exists()) for fn in key_files)
        if score > best_score:
            best_score = score
            best = d

    if best is not None and best_score > 0:
        return best, tried

    # if nothing matched, return primary (may be empty) with tried list
    return primary, tried


def _build_missing_list(snapshot_dir: Path, files: List[str]) -> List[str]:
    return [fn for fn in files if not (snapshot_dir / fn).exists()]


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Case-insensitive column match; return actual column name."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        k = str(c).lower()
        if k in lower_map:
            return lower_map[k]
    return None


def _safe_str_series(sr: pd.Series) -> pd.Series:
    """Normalize strings: NaN->'', strip, and remove common null-like tokens."""
    s = sr.fillna("").astype(str).str.strip()
    return s.replace(
        {
            "nan": "",
            "NaN": "",
            "<NA>": "",
            "<na>": "",
            "None": "",
            "NONE": "",
            "null": "",
            "NULL": "",
        }
    )


# -------------------------
# Normalization
# -------------------------
def _normalize_industry_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace("\u3000", " ")
    s = " ".join(s.split())
    for sep in ["/", "\\", "|", "｜", "-", "－", "—", "_"]:
        s = s.replace(sep, " ")
    return " ".join(s.split())


def _normalize_ts_code_one(x: str) -> str:
    """
    Normalize to:
      000001.SZ / 600000.SH / 830xxx.BJ (uppercase)
    Accept:
      000001
      000001.SZ / 000001sz / 000001.sz
      600000.SH
    """
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    s = s.replace(" ", "").upper()

    # has suffix
    if "." in s:
        left, right = s.split(".", 1)
        left = "".join([c for c in left if c.isdigit()])
        right = "".join([c for c in right if c.isalpha()])
        if len(left) == 6 and right in {"SZ", "SH", "BJ"}:
            return f"{left}.{right}"
        if len(left) == 6:
            if left.startswith(("6", "9")):
                return f"{left}.SH"
            if left.startswith(("8", "4")):
                return f"{left}.BJ"
            return f"{left}.SZ"
        return ""

    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) != 6:
        return ""
    if digits.startswith(("6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("8", "4")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _normalize_ts_code_series(sr: pd.Series) -> pd.Series:
    return _safe_str_series(sr).map(_normalize_ts_code_one)


def _normalize_table_ts_code(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ensure df['ts_code'] exists and is normalized.
    Returns (new_df, debug)
    """
    dbg: Dict[str, Any] = {"ok": False, "rows": 0, "code_col_used": None, "ts_code_nonempty": 0}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, dbg
    out = df.copy()
    dbg["rows"] = int(len(out))

    code_col = _first_existing_col(out, ["ts_code", "code", "TS_CODE", "股票代码", "证券代码"])
    if code_col:
        dbg["code_col_used"] = code_col
        out["ts_code"] = _normalize_ts_code_series(out[code_col])
    elif "ts_code" in out.columns:
        out["ts_code"] = _normalize_ts_code_series(out["ts_code"])
    else:
        out["ts_code"] = ""

    dbg["ts_code_nonempty"] = int((out["ts_code"].astype(str).str.strip() != "").sum())
    dbg["ok"] = True
    return out, dbg


def _normalize_industry_table(df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize a table to include:
      - ts_code (normalized)
      - industry (normalized)
    Keep original columns; add/overwrite standardized ones.
    """
    dbg: Dict[str, Any] = {
        "table": table_name,
        "rows": 0,
        "code_col_used": None,
        "industry_col_used": None,
        "ts_code_nonempty": 0,
        "industry_nonempty": 0,
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, dbg

    out, dbg_code = _normalize_table_ts_code(df)
    dbg["rows"] = dbg_code.get("rows", int(len(out))) if isinstance(out, pd.DataFrame) else 0
    dbg["code_col_used"] = dbg_code.get("code_col_used")
    dbg["ts_code_nonempty"] = dbg_code.get("ts_code_nonempty", 0)

    ind_col = _first_existing_col(
        out,
        [
            "industry",
            "行业",
            "所属行业",
            "所属板块",
            "板块",
            "板块名称",
            "行业名称",
            "申万行业",
            "sw_industry",
            "sw_industry_name",
            "concept",
            "题材",
        ],
    )
    if ind_col:
        dbg["industry_col_used"] = ind_col
        out["industry"] = _safe_str_series(out[ind_col]).map(_normalize_industry_name)
    else:
        out["industry"] = _safe_str_series(out.get("industry", pd.Series([], dtype=str))).map(_normalize_industry_name)

    # De-dup on ts_code: keep first non-empty industry
    if "ts_code" in out.columns:
        out["_ind_nonempty"] = (out["industry"].astype(str).str.strip() != "").astype(int)
        out = out.sort_values(by=["ts_code", "_ind_nonempty"], ascending=[True, False])
        out = out.drop_duplicates(subset=["ts_code"], keep="first")
        out = out.drop(columns=["_ind_nonempty"], errors="ignore").reset_index(drop=True)

    dbg["industry_nonempty"] = int((out["industry"].astype(str).str.strip() != "").sum())
    return out, dbg


def _apply_industry_map(universe: pd.DataFrame, src: pd.DataFrame, src_tag: str) -> Dict[str, Any]:
    """
    Fill universe industry from src[ts_code, industry], only where missing.
    """
    info: Dict[str, Any] = {"src": src_tag, "used": False, "map_size": 0, "filled": 0}
    if universe is None or universe.empty or src is None or src.empty:
        return info
    if "ts_code" not in src.columns or "industry" not in src.columns:
        return info

    mp_src = src[["ts_code", "industry"]].copy()
    mp_src["ts_code"] = _normalize_ts_code_series(mp_src["ts_code"])
    mp_src["industry"] = _safe_str_series(mp_src["industry"]).map(_normalize_industry_name)
    mp_src = mp_src[(mp_src["ts_code"] != "") & (mp_src["industry"] != "")]
    mp_src = mp_src.drop_duplicates(subset=["ts_code"], keep="first")

    mp = dict(zip(mp_src["ts_code"], mp_src["industry"]))
    info["map_size"] = int(len(mp))
    if not mp:
        return info

    if "行业" not in universe.columns:
        universe["行业"] = ""
    if "industry" not in universe.columns:
        universe["industry"] = ""

    before = int((_safe_str_series(universe["行业"]) != "").sum())

    need = _safe_str_series(universe["行业"]) == ""
    if need.any():
        filled_vals = universe.loc[need, "ts_code"].map(mp).fillna("")
        universe.loc[need, "行业"] = _safe_str_series(filled_vals)

    # normalize + sync
    universe["行业"] = _safe_str_series(universe["行业"]).map(_normalize_industry_name)
    universe["industry"] = _safe_str_series(universe["industry"]).map(_normalize_industry_name)
    sync = _safe_str_series(universe["industry"]) == ""
    universe.loc[sync, "industry"] = universe.loc[sync, "行业"]

    after = int((_safe_str_series(universe["行业"]) != "").sum())
    info["filled"] = int(after - before)
    info["used"] = True
    return info


# -------------------------
# Step0 Core
# -------------------------
def step0_build_universe(s: Settings, trade_date: str) -> Dict[str, Any]:
    """
    Build ctx with:
      - snapshot_dir & missing files
      - loaded tables
      - universe (guaranteed industry columns)
      - market metrics E1/E2/E3
      - debug_step0 (for diagnosing downstream issues)
    """
    # 1) select snapshot dir
    snap, snap_tried = _pick_snapshot_dir(s, trade_date)

    files_needed = [
        "daily.csv",
        "daily_basic.csv",
        "limit_list_d.csv",
        "limit_break_d.csv",
        "hot_boards.csv",
        "top_list.csv",
        "stock_basic.csv",
    ]
    snapshot_missing = _build_missing_list(snap, files_needed)

    # 2) read raw tables
    daily_raw = _read_csv_if_exists(snap / "daily.csv")
    daily_basic_raw = _read_csv_if_exists(snap / "daily_basic.csv")
    limit_list_d_raw = _read_csv_if_exists(snap / "limit_list_d.csv")
    limit_break_d_raw = _read_csv_if_exists(snap / "limit_break_d.csv")
    hot_boards_raw = _read_csv_if_exists(snap / "hot_boards.csv")
    top_list_raw = _read_csv_if_exists(snap / "top_list.csv")
    stock_basic_raw = _read_csv_if_exists(snap / "stock_basic.csv")

    # 3) normalize tables
    daily, dbg_daily_code = _normalize_table_ts_code(daily_raw)
    limit_list_d, dbg_limit_list_code = _normalize_table_ts_code(limit_list_d_raw)
    limit_break_d, dbg_limit_break_code = _normalize_table_ts_code(limit_break_d_raw)
    top_list, dbg_top_list_code = _normalize_table_ts_code(top_list_raw)

    stock_basic, dbg_stock_basic = _normalize_industry_table(stock_basic_raw, "stock_basic")
    daily_basic, dbg_daily_basic = _normalize_industry_table(daily_basic_raw, "daily_basic")

    # hot_boards: keep as-is (it is industry-centric table); still record basic shape
    hot_boards = hot_boards_raw.copy() if isinstance(hot_boards_raw, pd.DataFrame) else pd.DataFrame()

    # 4) build universe from daily
    universe = pd.DataFrame(columns=["ts_code", "name", "close", "pct_chg", "行业", "industry"])
    if isinstance(daily, pd.DataFrame) and not daily.empty:
        universe["ts_code"] = _normalize_ts_code_series(daily.get("ts_code", pd.Series([], dtype=str)))

        name_col = _first_existing_col(daily, ["name", "ts_name", "股票名称", "证券简称"])
        close_col = _first_existing_col(daily, ["close", "收盘价"])
        pct_col = _first_existing_col(daily, ["pct_chg", "pct_change", "change_pct", "涨跌幅"])

        if name_col:
            universe["name"] = _safe_str_series(daily[name_col])
        else:
            universe["name"] = ""

        if close_col:
            universe["close"] = _safe_str_series(daily[close_col])
        else:
            universe["close"] = ""

        if pct_col:
            universe["pct_chg"] = _safe_str_series(daily[pct_col])
        else:
            universe["pct_chg"] = ""
    else:
        # keep empty schema
        universe["ts_code"] = ""
        universe["name"] = ""
        universe["close"] = ""
        universe["pct_chg"] = ""

    universe["ts_code"] = _normalize_ts_code_series(universe["ts_code"])

    # 5) fill name from stock_basic if needed
    if not universe.empty:
        if int((_safe_str_series(universe["name"]) != "").sum()) == 0 and isinstance(stock_basic, pd.DataFrame) and not stock_basic.empty:
            sb_name = _first_existing_col(stock_basic, ["name", "ts_name", "证券简称", "股票名称"])
            if sb_name:
                mp = stock_basic[["ts_code", sb_name]].copy().rename(columns={sb_name: "_name"})
                mp["ts_code"] = _normalize_ts_code_series(mp["ts_code"])
                mp["_name"] = _safe_str_series(mp["_name"])
                mp = mp[(mp["ts_code"] != "") & (mp["_name"] != "")].drop_duplicates(subset=["ts_code"], keep="first")
                universe = universe.merge(mp, how="left", on="ts_code")
                universe["name"] = _safe_str_series(universe["name"])
                universe["_name"] = _safe_str_series(universe["_name"])
                mask = universe["name"] == ""
                universe.loc[mask, "name"] = universe.loc[mask, "_name"]
                universe.drop(columns=["_name"], inplace=True, errors="ignore")

    # 6) fill industry into universe: prefer stock_basic, then daily_basic
    if "行业" not in universe.columns:
        universe["行业"] = ""
    if "industry" not in universe.columns:
        universe["industry"] = ""

    dbg_industry_apply: Dict[str, Any] = {"stock_basic": {}, "daily_basic": {}}
    if isinstance(stock_basic, pd.DataFrame) and not stock_basic.empty:
        dbg_industry_apply["stock_basic"] = _apply_industry_map(universe, stock_basic, "stock_basic")
    if isinstance(daily_basic, pd.DataFrame) and not daily_basic.empty:
        dbg_industry_apply["daily_basic"] = _apply_industry_map(universe, daily_basic, "daily_basic")

    # 7) market (E1/E2/E3)
    e1 = 0
    if isinstance(limit_list_d, pd.DataFrame) and not limit_list_d.empty:
        if "ts_code" in limit_list_d.columns:
            e1 = int((_normalize_ts_code_series(limit_list_d["ts_code"]) != "").sum())
            # better: unique count
            e1 = int(_normalize_ts_code_series(limit_list_d["ts_code"]).nunique())
        else:
            e1 = int(len(limit_list_d))

    e2 = 0.0
    if isinstance(limit_break_d, pd.DataFrame) and not limit_break_d.empty:
        if "ts_code" in limit_break_d.columns:
            brk = int(_normalize_ts_code_series(limit_break_d["ts_code"]).nunique())
        else:
            brk = int(len(limit_break_d))
        denom = max(1, (e1 + brk))
        e2 = round(100.0 * brk / denom, 4)

    e3 = 0
    if isinstance(limit_list_d, pd.DataFrame) and not limit_list_d.empty:
        lb_col = _first_existing_col(limit_list_d, ["lbc", "lb", "consecutive", "连续涨停", "连板数", "连板高度"])
        if lb_col:
            try:
                e3 = int(pd.to_numeric(limit_list_d[lb_col], errors="coerce").fillna(0).max())
            except Exception:
                e3 = 0

    market = {"E1": int(e1), "E2": float(e2), "E3": int(e3)}

    # 8) debug
    debug_step0: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "snapshot_dir_tried": snap_tried,
        "snapshot_missing": snapshot_missing,
        "snapshot_has_all_core_files": bool(_dir_has_all_csv(snap, files_needed)) if snap.exists() else False,

        "rows": {
            "daily": int(len(daily)) if isinstance(daily, pd.DataFrame) else 0,
            "daily_basic": int(len(daily_basic)) if isinstance(daily_basic, pd.DataFrame) else 0,
            "limit_list_d": int(len(limit_list_d)) if isinstance(limit_list_d, pd.DataFrame) else 0,
            "limit_break_d": int(len(limit_break_d)) if isinstance(limit_break_d, pd.DataFrame) else 0,
            "hot_boards": int(len(hot_boards)) if isinstance(hot_boards, pd.DataFrame) else 0,
            "top_list": int(len(top_list)) if isinstance(top_list, pd.DataFrame) else 0,
            "stock_basic": int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0,
            "universe": int(len(universe)) if isinstance(universe, pd.DataFrame) else 0,
        },

        "universe_cols": list(universe.columns) if isinstance(universe, pd.DataFrame) else [],
        "universe_ts_code_nonempty": int((_safe_str_series(universe.get("ts_code", pd.Series([], dtype=str))) != "").sum())
        if isinstance(universe, pd.DataFrame) and not universe.empty
        else 0,
        "universe_industry_nonempty": int((_safe_str_series(universe.get("行业", pd.Series([], dtype=str))) != "").sum())
        if isinstance(universe, pd.DataFrame) and not universe.empty and ("行业" in universe.columns)
        else 0,

        "normalize": {
            "daily": dbg_daily_code,
            "limit_list_d": dbg_limit_list_code,
            "limit_break_d": dbg_limit_break_code,
            "top_list": dbg_top_list_code,
            "stock_basic": dbg_stock_basic,
            "daily_basic": dbg_daily_basic,
        },
        "industry_apply": dbg_industry_apply,
    }

    # 9) ctx
    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "snapshot_dir_tried": snap_tried,
        "snapshot_missing": snapshot_missing,
        "debug_step0": debug_step0,

        # tables
        "universe": universe,
        "daily": daily,
        "daily_basic": daily_basic,
        "limit_list_d": limit_list_d,
        "limit_break_d": limit_break_d,

        # compatibility keys
        "boards": hot_boards,
        "hot_boards": hot_boards,
        "dragon": top_list,
        "top_list": top_list,

        # industry master
        "stock_basic": stock_basic,

        "market": market,
    }
    return ctx


# Optional alias (some codebases call step0_run)
def run_step0(s: Settings, trade_date: str) -> Dict[str, Any]:
    return step0_build_universe(s, trade_date)
