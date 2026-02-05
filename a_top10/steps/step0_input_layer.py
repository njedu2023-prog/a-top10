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
    try:
        return pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        # 有些csv可能是gbk或混合，兜底再试一次
        try:
            return pd.read_csv(p, dtype=str, encoding="gbk")
        except Exception:
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
    """
    tried: List[str] = []

    # 1) 主推荐路径：由 Settings.data_repo.snapshot_dir 决定（你们系统主线）
    primary = s.data_repo.snapshot_dir(trade_date)
    tried.append(str(primary))
    if primary.exists() and primary.is_dir():
        return primary, tried

    # 2) 兜底：常见目录形态（兼容 snap/xxx.csv 这种旧约定）
    year = trade_date[:4]
    candidates: List[Path] = [
        Path("snap"),  # 例如：snap/daily.csv（不带日期子目录的老约定）
        Path("snap") / trade_date,  # snap/20260205/daily.csv
        Path("snapshots") / trade_date,  # snapshots/20260205/daily.csv
        Path("data_repo") / "snapshots" / trade_date,  # data_repo/snapshots/20260205/...
        Path("_warehouse") / "data" / "raw" / year / trade_date,  # _warehouse/data/raw/YYYY/YYYYMMDD
    ]

    key_files = [
        "daily.csv",
        "daily_basic.csv",
        "limit_list_d.csv",
        "limit_break_d.csv",
        "hot_boards.csv",
        "top_list.csv",
        "stock_basic.csv",
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
    daily_basic = _read_csv_if_exists(snap / "daily_basic.csv")
    limit_list_d = _read_csv_if_exists(snap / "limit_list_d.csv")
    limit_break_d = _read_csv_if_exists(snap / "limit_break_d.csv")
    hot_boards = _read_csv_if_exists(snap / "hot_boards.csv")
    top_list = _read_csv_if_exists(snap / "top_list.csv")
    stock_basic = _read_csv_if_exists(snap / "stock_basic.csv")

    # 4) 构造 Universe（统一底表）——尽量从 daily / stock_basic 补齐
    universe = pd.DataFrame(columns=["ts_code", "name", "close", "pct_chg"])

    if not daily.empty:
        cols = {str(c).lower(): c for c in daily.columns}
        ts_col = cols.get("ts_code", None) or cols.get("code", None)
        if ts_col:
            universe["ts_code"] = daily[ts_col].astype(str)

        name_col = _first_existing_col(daily, ["name", "ts_name", "股票名称"])
        close_col = _first_existing_col(daily, ["close", "收盘价"])
        pct_col = _first_existing_col(daily, ["pct_chg", "pct_change", "change_pct", "涨跌幅"])

        if name_col:
            universe["name"] = daily[name_col].astype(str)
        if close_col:
            universe["close"] = daily[close_col].astype(str)
        if pct_col:
            universe["pct_chg"] = daily[pct_col].astype(str)

    # 若 daily 没有 name，则用 stock_basic 尝试补（可选）
    if ("name" in universe.columns and universe["name"].isna().all()) or ("name" not in universe.columns):
        if (not stock_basic.empty) and ("ts_code" in universe.columns) and (not universe.empty):
            sb_ts = _first_existing_col(stock_basic, ["ts_code", "code"])
            sb_name = _first_existing_col(stock_basic, ["name"])
            if sb_ts and sb_name:
                tmp = stock_basic[[sb_ts, sb_name]].copy()
                tmp.columns = ["_ts", "_name"]
                tmp["_ts"] = tmp["_ts"].astype(str)
                tmp["_name"] = tmp["_name"].astype(str)
                universe = universe.merge(tmp, how="left", left_on="ts_code", right_on="_ts")
                universe["name"] = universe.get("name", pd.Series([""] * len(universe))).fillna(universe["_name"])
                universe.drop(columns=[c for c in ["_ts", "_name"] if c in universe.columns], inplace=True)

    # 5) market（E1/E2/E3）尽量从快照推断
    # E1 涨停家数：优先 limit_list_d 的去重 ts_code 行数
    e1 = 0
    if not limit_list_d.empty:
        ts_col = _first_existing_col(limit_list_d, ["ts_code", "code"])
        e1 = int(limit_list_d[ts_col].astype(str).nunique()) if ts_col else int(len(limit_list_d))

    # E2 炸板率（%）：若存在 limit_break_d（炸板池），用 炸板数 / (涨停数 + 炸板数)
    e2 = 0.0
    if not limit_break_d.empty:
        ts_col = _first_existing_col(limit_break_d, ["ts_code", "code"])
        brk = int(limit_break_d[ts_col].astype(str).nunique()) if ts_col else int(len(limit_break_d))
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

    # 6) ctx 输出（包含 debug 三项）
    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "snapshot_dir_tried": snap_tried,
        "snapshot_missing": snapshot_missing,
        # 原始表
        "universe": universe,
        "daily": daily,
        "daily_basic": daily_basic,
        "limit_list_d": limit_list_d,
        "limit_break_d": limit_break_d,
        "boards": hot_boards,
        "dragon": top_list,
        "stock_basic": stock_basic,
        "market": market,
    }
    return ctx
