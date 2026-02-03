# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml
import pandas as pd


# =========================================================
# DataRepo —— 给 step0 / step5 使用的真实数据仓库接口
# =========================================================
class DataRepo:
    """
    统一访问你们本地数据仓库结构。

    仓库结构（你给的真实路径）：
    _warehouse/
        a-share-top3-data/
            data/raw/
                2024/
                    20240102/
                        daily.csv
                        daily_basic.csv
                        limit_list_d.csv
                        hot_boards.csv
                        top_list.csv
    """

    def __init__(self, warehouse_root: str, repo_name: str, raw_dir: str):
        self.warehouse_root = Path(warehouse_root)
        self.repo_name = repo_name
        self.raw_dir = raw_dir

    def snapshot_dir(self, trade_date: str) -> Path:
        """返回某个交易日快照目录 Path"""
        year = trade_date[:4]
        return (
            self.warehouse_root
            / self.repo_name
            / self.raw_dir
            / year
            / trade_date
        )

    # ---------- 通用 CSV 读取 ----------
    @staticmethod
    def read_csv_if_exists(p: Path) -> pd.DataFrame:
        if not p.exists():
            return pd.DataFrame()
        for enc in ("utf-8", "gbk"):
            try:
                return pd.read_csv(p, dtype=str, encoding=enc)
            except Exception:
                pass
        return pd.DataFrame()

    # ---------- 常见数据表 ----------
    def read_daily(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "daily.csv")

    def read_daily_basic(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "daily_basic.csv")

    def read_limit_list(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "limit_list_d.csv")

    def read_hot_boards(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "hot_boards.csv")

    def read_top_list(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "top_list.csv")

    # ---------- Step5 训练闭环需要：列出全部 snapshot 日期 ----------
    def list_snapshot_dates(self) -> list[str]:
        """
        返回所有 YYYYMMDD 目录，供 Step5 训练使用。
        """
        root = self.warehouse_root / self.repo_name / self.raw_dir
        if not root.exists():
            return []

        dates = []
        for year_dir in root.iterdir():
            if not year_dir.is_dir():
                continue
            for d in year_dir.iterdir():
                if d.is_dir() and len(d.name) == 8 and d.name.isdigit():
                    dates.append(d.name)

        return sorted(dates)


# =========================================================
# 情绪闸门配置
# =========================================================
@dataclass
class EmotionGateCfg:
    min_limit_up_cnt: int = 50
    max_broken_rate: float = 0.35
    min_max连板高度: int = 3


# =========================================================
# IO / TopN 配置
# =========================================================
@dataclass
class IOCfg:
    outputs_dir: str = "outputs"
    keep_history: bool = True
    topn: int = 10
    topk_strength: int = 50
    candidate_size_hint: Tuple[int, int] = (30, 200)


# =========================================================
# 总配置（主入口 Settings）
# =========================================================
@dataclass
class DataRepoCfg:
    warehouse_root: str = "_warehouse"
    repo_name: str = "a-share-top3-data"
    raw_dir: str = "data/raw"


@dataclass
class Settings:
    version: str = "0.1"
    timezone: str = "Asia/Shanghai"

    # 关键：必须 factory，否则 Settings() 时自动创建独立对象
    data_repo: DataRepoCfg = field(default_factory=DataRepoCfg)
    io: IOCfg = field(default_factory=IOCfg)
    emotion_gate: EmotionGateCfg = field(default_factory=EmotionGateCfg)

    # 🟢 最关键修复：给所有 step 提供 DataRepo 实例
    def __post_init__(self):
        self.data_repo = DataRepo(
            warehouse_root=self.data_repo.warehouse_root,
            repo_name=self.data_repo.repo_name,
            raw_dir=self.data_repo.raw_dir,
        )

    def trade_date_resolver(self) -> str:
        td = os.getenv("TRADE_DATE", "").strip()
        if td:
            return td
        return datetime.now().strftime("%Y%m%d")


# =========================================================
# 配置加载（YAML）
# =========================================================
def load_settings(config_path: str) -> Settings:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    raw: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    s = Settings()

    # 基础字段
    s.version = str(raw.get("version", s.version))
    s.timezone = str(raw.get("timezone", s.timezone))

    # -------- data_repo --------
    dr = raw.get("data_repo", {}) or {}
    s.data_repo = DataRepoCfg(
        warehouse_root=str(dr.get("warehouse_root", s.data_repo.warehouse_root)),
        repo_name=str(dr.get("repo_name", s.data_repo.repo_name)),
        raw_dir=str(dr.get("raw_dir", s.data_repo.raw_dir)),
    )

    # 重要：重新生成 DataRepo 实例
    s.data_repo = DataRepo(
        warehouse_root=s.data_repo.warehouse_root,
        repo_name=s.data_repo.repo_name,
        raw_dir=s.data_repo.raw_dir,
    )

    # -------- io --------
    io_raw = raw.get("io", {}) or {}
    hint = io_raw.get("candidate_size_hint", list(s.io.candidate_size_hint))
    if isinstance(hint, (list, tuple)) and len(hint) == 2:
        hint = (int(hint[0]), int(hint[1]))

    s.io = IOCfg(
        outputs_dir=str(io_raw.get("outputs_dir", s.io.outputs_dir)),
        keep_history=bool(io_raw.get("keep_history", s.io.keep_history)),
        topn=int(io_raw.get("topn", s.io.topn)),
        topk_strength=int(io_raw.get("topk_strength", s.io.topk_strength)),
        candidate_size_hint=hint,
    )

    # -------- emotion_gate --------
    filters = raw.get("filters", {}) or {}
    eg = filters.get("emotion_gate", {}) or {}
    s.emotion_gate = EmotionGateCfg(
        min_limit_up_cnt=int(eg.get("min_limit_up_cnt", s.emotion_gate.min_limit_up_cnt)),
        max_broken_rate=float(eg.get("max_broken_rate", s.emotion_gate.max_broken_rate)),
        min_max连板高度=int(eg.get("min_max连板高度", s.emotion_gate.min_max连板高度)),
    )

    return s
