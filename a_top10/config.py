from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class DataRepoCfg:
    warehouse_root: str = "_warehouse"
    repo_name: str = "a-share-top3-data"
    raw_dir: str = "data/raw"

    def snapshot_dir(self, trade_date: str) -> Path:
        # _warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD/
        year = trade_date[:4]
        return Path(self.warehouse_root) / self.repo_name / self.raw_dir / year / trade_date


@dataclass
class EmotionGateCfg:
    min_limit_up_cnt: int = 50
    max_broken_rate: float = 0.35
    min_max连板高度: int = 3


@dataclass
class IOCfg:
    outputs_dir: str = "outputs"
    keep_history: bool = True
    topn: int = 10
    topk_strength: int = 50
    candidate_size_hint: Tuple[int, int] = (30, 200)


@dataclass
class Settings:
    version: str = "0.1"
    timezone: str = "Asia/Shanghai"
    data_repo: DataRepoCfg = DataRepoCfg()
    io: IOCfg = IOCfg()
    emotion_gate: EmotionGateCfg = EmotionGateCfg()

    # 先给最小实现：trade_date 解析（ENV优先，其次今天）
    def trade_date_resolver(self) -> str:
        td = os.getenv("TRADE_DATE", "").strip()
        if td:
            return td
        return datetime.now().strftime("%Y%m%d")


def load_settings(config_path: str) -> Settings:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    raw: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    s = Settings()
    s.version = str(raw.get("version", s.version))
    s.timezone = str(raw.get("timezone", s.timezone))

    # data_repo
    dr = raw.get("data_repo", {}) or {}
    s.data_repo = DataRepoCfg(
        warehouse_root=str(dr.get("warehouse_root", s.data_repo.warehouse_root)),
        repo_name=str(dr.get("repo_name", s.data_repo.repo_name)),
        raw_dir=str(dr.get("raw_dir", s.data_repo.raw_dir)),
    )

    # io
    io = raw.get("io", {}) or {}
    hint = io.get("candidate_size_hint", list(s.io.candidate_size_hint))
    if isinstance(hint, (list, tuple)) and len(hint) == 2:
        hint = (int(hint[0]), int(hint[1]))
    else:
        hint = s.io.candidate_size_hint

    s.io = IOCfg(
        outputs_dir=str(io.get("outputs_dir", s.io.outputs_dir)),
        keep_history=bool(io.get("keep_history", s.io.keep_history)),
        topn=int(io.get("topn", s.io.topn)),
        topk_strength=int(io.get("topk_strength", s.io.topk_strength)),
        candidate_size_hint=hint,
    )

    # filters.emotion_gate
    filters = raw.get("filters", {}) or {}
    eg = (filters.get("emotion_gate", {}) or {})
    s.emotion_gate = EmotionGateCfg(
        min_limit_up_cnt=int(eg.get("min_limit_up_cnt", s.emotion_gate.min_limit_up_cnt)),
        max_broken_rate=float(eg.get("max_broken_rate", s.emotion_gate.max_broken_rate)),
        min_max连板高度=int(eg.get("min_max连板高度", s.emotion_gate.min_max连板高度)),
    )

    return s
