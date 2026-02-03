#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 Top3 的本地行情快照复制到本项目 a-top10 的 data_repo/snapshots 下，
让主程序 step0~step6 能正常读取。
"""

from __future__ import annotations
import shutil
from pathlib import Path

# 你的 Top3 数据仓库快照根目录（本地路径）
LOCAL_SNAPSHOT_ROOT = Path("/Users/xujing/Documents/TopusMlData/snapshots")

# 目标：a-top10 项目的 snapshot 目录（相对当前项目）
TARGET_SNAPSHOT_ROOT = Path(__file__).resolve().parent.parent / "data_repo" / "snapshots"
TARGET_SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)


def pull_one_snapshot(trade_date: str) -> str:
    """
    trade_date: '20260204'

    从 Top3 仓库复制 trade_date 的快照目录到本项目 data_repo/snapshots 下
    """

    src = LOCAL_SNAPSHOT_ROOT / trade_date
    dst = TARGET_SNAPSHOT_ROOT / trade_date

    if not src.exists():
        return f"[ERROR] 源快照不存在：{src}"

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    return f"[OK] {trade_date} 快照已复制到 {dst}"


def pull_latest() -> str:
    """
    自动找到本地最新快照，并复制
    """
    all_dirs = [d for d in LOCAL_SNAPSHOT_ROOT.iterdir() if d.is_dir() and len(d.name) == 8]
    if not all_dirs:
        return f"[ERROR] 本地快照目录为空：{LOCAL_SNAPSHOT_ROOT}"

    latest = sorted(all_dirs)[-1].name
    return pull_one_snapshot(latest)


if __name__ == "__main__":
    print(pull_latest())
