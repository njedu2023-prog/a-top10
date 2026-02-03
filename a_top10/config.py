from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


# =========================
# 数据仓库配置
# =========================
@dataclass
class DataRepoCfg:
    warehouse_root: str = "_warehouse"
    repo_name: str = "a-share-top3-data"
    raw_dir: str = "data/raw"

    def snapshot_dir(self, trade_date: str) -> Path:
        # _warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD/
        year = trade_date[:4]
        return (
            Path(self.warehouse_root)
            / self.repo_name
            / self.raw_dir
            / year
            / trade_date
        )


# =========================
# 情绪闸门配置
# =========================
@dataclass
class EmotionGateCfg:
    min_limit_up_cnt: int = 50
    max_broken_rate: float = 0.35
    min_max连板高度: int = 3


# =========================
# IO / TopN 配置
# =========================
@dataclass
class IOCfg:
    outputs_dir: str = "outputs"
    keep_history: bool = True
    topn: int = 10
    topk_strength: int = 50
    candidate_size_hint: Tuple[int, int] = (30, 200)


# =========================
# 总配置
# =========================
@dataclass
class Settings:
    version: str = "0.1"
    timezone: str = "Asia/Shanghai"

    # ⚠️ 关键修复点：全部使用 default_factory
    data_repo: DataRepoCfg = field(default_factory=DataRepoCfg)
    io: IOCfg = field(default_factory=IOCfg)
    emotion_gate: EmotionGateCfg = field(default_factory=EmotionGateCfg)

    def trade_date_resolver(self) -> str:
        """
        交易日解析顺序：
        1. 环境变量 TRADE_DATE
        2. 今天（YYYYMMDD）
        """
        td = os.getenv("TRADE_DATE", "").strip()
        if td:
            return td
        return datetime.now().strftime("%Y%m%d")


# =========================
# 配置加载
# =========================
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

    # -------- io --------
    io_raw = raw.get("io", {}) or {}
    hint = io_raw.get("candidate_size_hint", list(s.io.candidate_size_hint))
    if isinstance(hint, (list, tuple)) and len(hint) == 2:
        hint = (int(hint[0]), int(hint[1]))
    else:
        hint = s.io.candidate_size_hint

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
        min_limit_up_cnt=int(
            eg.get("min_limit_up_cnt", s.emotion_gate.min_limit_up_cnt)
        ),
        max_broken_rate=float(
            eg.get("max_broken_rate", s.emotion_gate.max_broken_rate)
        ),
        min_max连板高度=int(
            eg.get("min_max连板高度", s.emotion_gate.min_max连板高度)
        ),
    )

    return s
