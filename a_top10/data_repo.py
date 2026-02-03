# -*- coding: utf-8 -*-
"""
DataRepo: 统一访问你们本地数据仓库结构

约定目录结构：
data_root/
    snapshots/
        20240101/
            daily.csv
            daily_basic.csv
            limit_list_d.csv
            hot_boards.csv
            top_list.csv
        20240102/
            ...
"""

from __future__ import annotations
from pathlib import Path
from typing import List


class DataRepo:
    """
    负责读取 snapshot 目录、列出可用日期，提供 snapshot_dir(date)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_root = self.root / "snapshots"

    # ---- 返回某一天的 snapshot 目录 ----
    def snapshot_dir(self, trade_date: str) -> Path:
        """
        返回形如 root/snapshots/20240101 的路径
        """
        return self.snapshots_root / trade_date

    # ---- 列出所有可用的 snapshot 日期 ----
    def list_snapshot_dates(self) -> List[str]:
        """
        返回所有日期（目录名），例如 ["20240101", "20240102", ...]
        """
        if not self.snapshots_root.exists():
            return []

        dates = []
        for p in self.snapshots_root.iterdir():
            if p.is_dir() and len(p.name) >= 8 and p.name.isdigit():
                dates.append(p.name)

        return sorted(dates)

    # ---- 模型目录（供 step5 使用）----
    @property
    def models_dir(self) -> Path:
        """
        例如：root/models/
        """
        p = self.root / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p
