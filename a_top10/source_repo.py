# -*- coding: utf-8 -*-
"""
source_repo.py
负责告诉 pull_snapshot.py：数据仓库在哪里
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

class SourceRepo:
    """
    指向你的 Top3 数据仓库，例如：
    a_top10/snapshots/20260203/daily.csv
    """

    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser().resolve()

        if not self.root.exists():
            raise FileNotFoundError(f"数据仓库不存在: {self.root}")

    # -------------------------------------------------------
    # 返回某一天的 snapshot 目录，例如：
    #   root/snapshots/20260203/
    # -------------------------------------------------------
    def snapshot_dir(self, trade_date: str) -> Path:
        snap = self.root / "snapshots" / trade_date
        return snap

    # -------------------------------------------------------
    # 列出所有 snapshot 日期目录（用于 Step5 训练）
    # -------------------------------------------------------
    def list_snapshot_dates(self):
        snap_root = self.root / "snapshots"
        if not snap_root.exists():
            return []

        return sorted(
            [p.name for p in snap_root.iterdir() if p.is_dir()]
        )
