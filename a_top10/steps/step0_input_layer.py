from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


# -------------------------
# IO Utils
# -------------------------
def _read_csv_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    # 依次尝试多种编码（快照经常是 utf-8-sig / gbk）
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _dir_has_any_csv(d: Path, filenames: List[str]) -> bool:
    if not d.exists() or (not d.is_dir()):
        return False
    for fn in filenames:
        if (d / fn).exists():
            return True
    return False


def _pick_snapshot_dir(s: Settings, trade_date: str) -> Tuple[Path, List[str]]:
    """
    选择一个最可能的快照目录：
    - 优先：s.data_repo.snapshot_dir(trade_date)
    - 兜底：尝试一组常见目录（不写死具体仓库名，只写“形态”）
    返回：
      chosen_dir, tried_dirs(str list)

    ✅ 修复：不能只看目录存在，必须至少命中关键 CSV，否则会选到空目录，导致下游全空。
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

    # 1) 主推荐路径：由 Settings.data_repo.snapshot_dir 决定（你们系统主线）
    primary = s.data_repo.snapshot_dir(trade_date)
    tried.append(str(primary))
    if primary.exists() and primary.is_dir() and _dir_has_any_csv(primary, key_files):
        return primary, tried

    # 2) 兜底：常见目录形态
    year = trade_date[:4]

    repo_name = None
    try:
        repo_name = getattr(getattr(s, "data_repo", None), "name", None)
    except Exception:
        repo_name = None
    if not repo_name:
        repo_name = "a-share-top3-data"

    candidates: List[Path] = [
        Path("snap"),
        Path("snap") / trade_date,
        Path("snapshots") / trade_date,
        Path("data_repo") / "snapshots" / trade_date,
        Path("_warehouse") / repo_name / "data" / "raw" / year / trade_date,
        Path("_warehouse") / repo_name / "raw" / year / trade_date,
        Path("_warehouse") / "data" / "raw" / year / trade_date,
    ]

    for d in candidates:
        tried.append(str(d))
        if _dir_has_any_csv(d, key_files):
            return d, tried

    # 3) 实在找不到：仍返回 primary（保持与主线一致），但会在 ctx 里标记 missing
    return primary, tried


def _build_missing_list(snapshot_dir: Path, files: List[str]) -> List[str]:
    missing: List[str] = []
    for fn in files:
        if not (snapshot_dir / fn).exists():
            missing.append(fn)
    return missing


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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
    """
    统一清洗：
    - NaN -> ""
    - "nan"/"<na>"/"None" 等字符串 -> ""
    - strip
    """
    s = sr.fillna("").astype(str).str.strip()
    s = s.replace(
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
    return s


def _normalize_industry_table(df: pd.DataFrame, *, table_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    关键修复：把“行业/板块”相关字段统一规范为 df["industry"]，供 Step4 直接读取。
    同时确保存在 df["ts_code"]（如果只有 code，也会补一列 ts_code=code）。

    返回：(new_df, debug_info)
    """
    dbg: Dict[str, Any] = {
        "table": table_name,
        "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "has_ts_code": False,
        "code_col_used": None,
        "industry_col_used": None,
        "industry_nonempty": 0,
    }

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, dbg

    out = df.copy()

    # 1) 统一 code 列：保证 ts_code 存在
    code_col = _first_existing_col(out, ["ts_code", "code", "股票代码", "证券代码", "TS_CODE"])
    if code_col:
        dbg["code_col_used"] = code_col
        if "ts_code" not in out.columns:
            out["ts_code"] = _safe_str_series(out[code_col])
        else:
            out["ts_code"] = _safe_str_series(out["ts_code"])
        dbg["has_ts_code"] = True
    else:
        # 仍然保证 ts_code 列存在，避免下游 KeyError
        if "ts_code" not in out.columns:
            out["ts_code"] = ""

    # 2) 统一行业列：industry
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
        out["industry"] = _safe_str_series(out[ind_col])
    else:
        if "industry" not in out.columns:
            out["industry"] = ""
        else:
            out["industry"] = _safe_str_series(out["industry"])

    dbg["industry_nonempty"] = int((out["industry"].astype(str).str.strip() != "").sum())
    return out, dbg


# -------------------------
# Step0 Core
# -------------------------
def step0_build_universe(s: Settings, trade_date: str) -> Dict[str, Any]:
    """
    Step0: Input Layer
    - 负责从“数据仓库快照目录”读入各类原始 CSV
    - 产出 ctx（后续 step1~step6 都从 ctx 取数据）
    - 输出关键 debug 字段：
        snapshot_dir / snapshot_dir_tried / snapshot_missing
    """
    # 1) 选择快照目录（并记录尝试路径）
    snap, snap_tried = _pick_snapshot_dir(s, trade_date)

    # 2) 定义要读取的文件集合（与 TOP3 数据仓库快照对齐的常见集合）
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

    # 3) 读取数据（缺了就空表，不报错）
    daily = _read_csv_if_exists(snap / "daily.csv")
    daily_basic_raw = _read_csv_if_exists(snap / "daily_basic.csv")
    limit_list_d = _read_csv_if_exists(snap / "limit_list_d.csv")
    limit_break_d = _read_csv_if_exists(snap / "limit_break_d.csv")
    hot_boards = _read_csv_if_exists(snap / "hot_boards.csv")
    top_list = _read_csv_if_exists(snap / "top_list.csv")
    stock_basic_raw = _read_csv_if_exists(snap / "stock_basic.csv")

    # ✅ 关键修复：规范化 stock_basic / daily_basic 的 industry/ts_code，供 Step4 使用
    stock_basic, dbg_stock_basic = _normalize_industry_table(stock_basic_raw, table_name="stock_basic")
    daily_basic, dbg_daily_basic = _normalize_industry_table(daily_basic_raw, table_name="daily_basic")

    # 4) 构造 Universe（统一底表）——尽量从 daily / stock_basic 补齐
    universe = pd.DataFrame(columns=["ts_code", "name", "close", "pct_chg"])

    if not daily.empty:
        cols = {str(c).lower(): c for c in daily.columns}
        ts_col = cols.get("ts_code", None) or cols.get("code", None) or cols.get("ts_code", None)
        if ts_col:
            universe["ts_code"] = _safe_str_series(daily[ts_col])

        name_col = _first_existing_col(daily, ["name", "ts_name", "股票名称", "证券简称"])
        close_col = _first_existing_col(daily, ["close", "收盘价"])
        pct_col = _first_existing_col(daily, ["pct_chg", "pct_change", "change_pct", "涨跌幅"])

        if name_col:
            universe["name"] = _safe_str_series(daily[name_col])
        if close_col:
            universe["close"] = _safe_str_series(daily[close_col])
        if pct_col:
            universe["pct_chg"] = _safe_str_series(daily[pct_col])

    # 若 daily 没有 name（或全空），则用 stock_basic 尝试补
    if "name" not in universe.columns:
        universe["name"] = ""
    name_all_blank = True
    if not universe.empty:
        name_all_blank = int((_safe_str_series(universe["name"]) != "").sum()) == 0

    if name_all_blank:
        if (not stock_basic.empty) and ("ts_code" in universe.columns) and (not universe.empty):
            sb_ts = _first_existing_col(stock_basic, ["ts_code", "code"])
            sb_name = _first_existing_col(stock_basic, ["name", "ts_name", "证券简称", "股票名称"])
            if sb_ts and sb_name:
                tmp = stock_basic[[sb_ts, sb_name]].copy()
                tmp.columns = ["_ts", "_name"]
                tmp["_ts"] = _safe_str_series(tmp["_ts"])
                tmp["_name"] = _safe_str_series(tmp["_name"])
                tmp = tmp.drop_duplicates(subset=["_ts"], keep="first")

                universe = universe.merge(tmp, how="left", left_on="ts_code", right_on="_ts")
                universe["name"] = _safe_str_series(universe.get("name", pd.Series([""] * len(universe))))
                universe["_name"] = _safe_str_series(universe.get("_name", pd.Series([""] * len(universe))))
                mask = universe["name"].astype(str).str.strip() == ""
                universe.loc[mask, "name"] = universe.loc[mask, "_name"]
                universe.drop(columns=[c for c in ["_ts", "_name"] if c in universe.columns], inplace=True)

    # ✅ 关键增强：尽量把 “行业/板块/industry” merge 进 universe（供 Step2/Step3 侧使用）
    def _merge_industry_from(df_src: pd.DataFrame) -> None:
        nonlocal universe
        if df_src is None or df_src.empty or universe is None or universe.empty:
            return

        u_ts = _first_existing_col(universe, ["ts_code", "code"])
        if not u_ts:
            return

        src_ts = _first_existing_col(df_src, ["ts_code", "code"])
        src_ind = _first_existing_col(df_src, ["industry", "行业", "板块", "所属行业", "所属板块", "行业名称", "板块名称"])
        if not (src_ts and src_ind):
            return

        tmp = df_src[[src_ts, src_ind]].copy()
        tmp.columns = ["_ts", "_industry"]
        tmp["_ts"] = _safe_str_series(tmp["_ts"])
        tmp["_industry"] = _safe_str_series(tmp["_industry"])
        tmp = tmp[(tmp["_ts"] != "")].drop_duplicates(subset=["_ts"], keep="first")

        universe = universe.merge(tmp, how="left", left_on=u_ts, right_on="_ts")

        if "行业" not in universe.columns:
            universe["行业"] = ""
        universe["行业"] = _safe_str_series(universe["行业"])
        universe["_industry"] = _safe_str_series(universe["_industry"])

        # 只在行业为空时补
        mask = universe["行业"].astype(str).str.strip() == ""
        universe.loc[mask, "行业"] = universe.loc[mask, "_industry"]

        universe.drop(columns=[c for c in ["_ts", "_industry"] if c in universe.columns], inplace=True)

    if not stock_basic.empty:
        _merge_industry_from(stock_basic)
    if not daily_basic.empty:
        _merge_industry_from(daily_basic)

    # 5) market（E1/E2/E3）尽量从快照推断
    # E1 涨停家数：优先 limit_list_d 的去重 ts_code 行数
    e1 = 0
    if not limit_list_d.empty:
        ts_col = _first_existing_col(limit_list_d, ["ts_code", "code", "TS_CODE", "股票代码", "证券代码"])
        if ts_col:
            e1 = int(_safe_str_series(limit_list_d[ts_col]).nunique())
        else:
            e1 = int(len(limit_list_d))

    # E2 炸板率（%）：若存在 limit_break_d（炸板池），用 炸板数 / (涨停数 + 炸板数)
    e2 = 0.0
    if not limit_break_d.empty:
        ts_col = _first_existing_col(limit_break_d, ["ts_code", "code", "TS_CODE", "股票代码", "证券代码"])
        if ts_col:
            brk = int(_safe_str_series(limit_break_d[ts_col]).nunique())
        else:
            brk = int(len(limit_break_d))
        denom = max(1, (e1 + brk))
        e2 = round(100.0 * brk / denom, 4)

    # E3 连板高度：尝试从 limit_list_d 的常见字段取 max
    e3 = 0
    if not limit_list_d.empty:
        lb_col = _first_existing_col(limit_list_d, ["lbc", "lb", "consecutive", "连续涨停", "连板数", "连板高度"])
        if lb_col:
            try:
                e3 = int(pd.to_numeric(limit_list_d[lb_col], errors="coerce").fillna(0).max())
            except Exception:
                e3 = 0

    market = {"E1": int(e1), "E2": float(e2), "E3": int(e3)}

    # ✅ debug：用行数确认到底有没有读到数据（定位题材加成为 0 的根因）
    debug_tables = {
        "rows_daily": int(len(daily)) if isinstance(daily, pd.DataFrame) else 0,
        "rows_daily_basic": int(len(daily_basic)) if isinstance(daily_basic, pd.DataFrame) else 0,
        "rows_limit_list_d": int(len(limit_list_d)) if isinstance(limit_list_d, pd.DataFrame) else 0,
        "rows_limit_break_d": int(len(limit_break_d)) if isinstance(limit_break_d, pd.DataFrame) else 0,
        "rows_hot_boards": int(len(hot_boards)) if isinstance(hot_boards, pd.DataFrame) else 0,
        "rows_top_list": int(len(top_list)) if isinstance(top_list, pd.DataFrame) else 0,
        "rows_stock_basic": int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0,
        "rows_universe": int(len(universe)) if isinstance(universe, pd.DataFrame) else 0,
        "universe_has_industry": bool(
            isinstance(universe, pd.DataFrame)
            and ("行业" in universe.columns or "板块" in universe.columns or "industry" in universe.columns)
        ),
        # ✅ 行业链路关键诊断（给 Step4 用）
        "stock_basic_norm": dbg_stock_basic,
        "daily_basic_norm": dbg_daily_basic,
        "universe_industry_nonempty": int((_safe_str_series(universe.get("行业", pd.Series([], dtype=str))) != "").sum())
        if isinstance(universe, pd.DataFrame) and (not universe.empty) and ("行业" in universe.columns)
        else 0,
    }

    # 6) ctx 输出（包含 debug 三项）
    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "snapshot_dir_tried": snap_tried,
        "snapshot_missing": snapshot_missing,
        "debug_step0": debug_tables,

        # 原始表
        "universe": universe,
        "daily": daily,
        "daily_basic": daily_basic,  # ✅ 写回规范化后的 daily_basic
        "limit_list_d": limit_list_d,
        "limit_break_d": limit_break_d,

        # ✅ 同时写两个 key：兼容不同 step 的读取习惯
        "boards": hot_boards,
        "hot_boards": hot_boards,

        "dragon": top_list,
        "top_list": top_list,

        "stock_basic": stock_basic,  # ✅ 写回规范化后的 stock_basic（关键！供 Step4 补齐行业）
        "market": market,
    }
    return ctx
