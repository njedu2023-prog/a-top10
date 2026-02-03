from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from a_top10.config import Settings


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


def step0_build_universe(s: Settings, trade_date: str) -> Dict[str, Any]:
    snap = s.data_repo.snapshot_dir(trade_date)

    # 你们数据仓库常见文件：daily.csv / daily_basic.csv / limit_list_d.csv / hot_boards.csv 等
    daily = _read_csv_if_exists(snap / "daily.csv")
    daily_basic = _read_csv_if_exists(snap / "daily_basic.csv")
    limit_list_d = _read_csv_if_exists(snap / "limit_list_d.csv")
    hot_boards = _read_csv_if_exists(snap / "hot_boards.csv")
    top_list = _read_csv_if_exists(snap / "top_list.csv")  # 龙虎榜（如存在）

    # ---- 构造 Universe（统一底表）----
    # V0.1：只保证最小字段存在：ts_code / name（若有）/ close / pct_chg
    # daily 为空也没关系，输出空表，但字段固定
    universe = pd.DataFrame(columns=["ts_code", "name", "close", "pct_chg"])
    if not daily.empty:
        cols = {c.lower(): c for c in daily.columns}
        ts_col = cols.get("ts_code", None)
        if ts_col:
            universe["ts_code"] = daily[ts_col].astype(str)
        if "name" in daily.columns:
            universe["name"] = daily["name"].astype(str)
        if "close" in daily.columns:
            universe["close"] = daily["close"].astype(str)
        if "pct_chg" in daily.columns:
            universe["pct_chg"] = daily["pct_chg"].astype(str)

    # ---- market（E1/E2/E3）----
    # V0.1 先占位：如果 limit_list_d 有，则用它算 E1（涨停家数）；E2/E3 暂置0
    E1 = int(len(limit_list_d)) if not limit_list_d.empty else 0
    market = {"E1": E1, "E2": 0.0, "E3": 0}

    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "universe": universe,
        "daily": daily,
        "daily_basic": daily_basic,
        "limit_list_d": limit_list_d,
        "boards": hot_boards,
        "dragon": top_list,
        "market": market,
    }
    return ctx
