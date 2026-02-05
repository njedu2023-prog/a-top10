from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


def _read_csv_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(p, dtype=str, encoding="gbk")
        except Exception:
            return pd.DataFrame()


def _exists_nonempty_dir(p: Path) -> bool:
    try:
        return p.exists() and p.is_dir()
    except Exception:
        return False


def _try_find_snapshot_dir(trade_date: str, primary: Path) -> Tuple[Path, List[str]]:
    """
    返回：命中的快照目录 + 尝试过的候选目录（用于debug）
    """
    tried: List[str] = []

    # 1) 主路径（来自 Settings）
    candidates: List[Path] = [primary]

    # 2) 常见兼容路径（按你们历史讨论补齐）
    year = trade_date[:4]
    candidates += [
        Path("snap") / trade_date,
        Path("snapshots") / trade_date,
        Path("data_repo") / "snapshots" / trade_date,
        Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / year / trade_date,
        Path("_warehouse") / "a_share_top3_data" / "data" / "raw" / year / trade_date,
    ]

    # 去重保持顺序
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        cs = str(c)
        if cs not in seen:
            seen.add(cs)
            uniq.append(c)

    # 先按候选目录直接命中
    for d in uniq:
        tried.append(str(d))
        if _exists_nonempty_dir(d):
            # 只要目录存在，并且至少有一个关键文件存在，就认为命中
            if (d / "daily.csv").exists() or (d / "daily_basic.csv").exists() or (d / "limit_list_d.csv").exists():
                return d, tried

    # 3) 轻量 glob：找 **/YYYYMMDD/daily.csv
    #    只在仓库根目录下搜一次，避免太重
    pattern = f"**/{trade_date}/daily.csv"
    hits = list(Path(".").glob(pattern))
    if hits:
        d = hits[0].parent
        tried.append(f"glob:{pattern}->{d}")
        return d, tried

    # 4) 兜底：返回 primary（让上层显示缺失）
    return primary, tried


def step0_build_universe(s: Settings, trade_date: str) -> Dict[str, Any]:
    # Settings 给的主路径
    primary = s.data_repo.snapshot_dir(trade_date)
    snap, tried = _try_find_snapshot_dir(trade_date, primary)

    # 你们数据仓库常见文件：daily.csv / daily_basic.csv / limit_list_d.csv / hot_boards.csv 等
    daily = _read_csv_if_exists(snap / "daily.csv")
    daily_basic = _read_csv_if_exists(snap / "daily_basic.csv")
    limit_list_d = _read_csv_if_exists(snap / "limit_list_d.csv")
    hot_boards = _read_csv_if_exists(snap / "hot_boards.csv")
    top_list = _read_csv_if_exists(snap / "top_list.csv")  # 龙虎榜（如存在）

    # ---- 构造 Universe（统一底表）----
    universe = pd.DataFrame(columns=["ts_code", "name", "close", "pct_chg"])
    if not daily.empty:
        cols = {str(c).lower(): c for c in daily.columns}
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
    E1 = int(len(limit_list_d)) if not limit_list_d.empty else 0
    market = {"E1": E1, "E2": 0.0, "E3": 0}

    # ---- 快照可观测性（关键）----
    file_checks = {
        "daily.csv": {"exists": (snap / "daily.csv").exists(), "rows": int(len(daily))},
        "daily_basic.csv": {"exists": (snap / "daily_basic.csv").exists(), "rows": int(len(daily_basic))},
        "limit_list_d.csv": {"exists": (snap / "limit_list_d.csv").exists(), "rows": int(len(limit_list_d))},
        "hot_boards.csv": {"exists": (snap / "hot_boards.csv").exists(), "rows": int(len(hot_boards))},
        "top_list.csv": {"exists": (snap / "top_list.csv").exists(), "rows": int(len(top_list))},
    }
    missing = [k for k, v in file_checks.items() if not v["exists"]]

    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "snapshot_dir_tried": tried,  # 你一眼能看出它找过哪些目录
        "snapshot_files": file_checks,
        "snapshot_missing": missing,

        "universe": universe,
        "daily": daily,
        "daily_basic": daily_basic,
        "limit_list_d": limit_list_d,
        "boards": hot_boards,   # Step4 会从这里取
        "dragon": top_list,     # Step4 会从这里取
        "market": market,
    }
    return ctx
