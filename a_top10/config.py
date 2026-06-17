# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from types import SimpleNamespace


# =========================================================
# A 股交易日历兜底
# =========================================================
# 说明：
# 1. 专业交易日历的第一优先级仍应来自数据仓库/外部 trade_cal。
# 2. 这里提供硬兜底，避免节假日前后错误回退到“自然工作日”。
# 3. 2026 年休市安排按上交所公告口径写入，用于修复 20260430 -> 20260506。
A_SHARE_HOLIDAY_RANGES: Tuple[Tuple[str, str], ...] = (
    # 2024
    ("20240101", "20240101"),
    ("20240209", "20240218"),
    ("20240404", "20240406"),
    ("20240501", "20240505"),
    ("20240608", "20240610"),
    ("20240915", "20240917"),
    ("20241001", "20241007"),
    # 2025
    ("20250101", "20250101"),
    ("20250128", "20250204"),
    ("20250404", "20250406"),
    ("20250501", "20250505"),
    ("20250531", "20250602"),
    ("20251001", "20251008"),
    # 2026
    ("20260101", "20260103"),
    ("20260215", "20260223"),
    ("20260404", "20260406"),
    ("20260501", "20260505"),
    ("20260619", "20260621"),
    ("20260925", "20260927"),
    ("20261001", "20261007"),
)


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(str(s), "%Y%m%d").date()


def _fmt_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _date_range(start: str, end: str) -> set[str]:
    s = _parse_yyyymmdd(start)
    e = _parse_yyyymmdd(end)
    out: set[str] = set()
    cur = s
    while cur <= e:
        out.add(_fmt_yyyymmdd(cur))
        cur += timedelta(days=1)
    return out


A_SHARE_HOLIDAYS: set[str] = set()
for _s, _e in A_SHARE_HOLIDAY_RANGES:
    A_SHARE_HOLIDAYS.update(_date_range(_s, _e))


def is_a_share_trading_day(trade_date: str) -> bool:
    """A 股交易日判断：周末 + 法定休市日过滤。"""
    try:
        d = _parse_yyyymmdd(trade_date)
    except Exception:
        return False
    if d.weekday() >= 5:
        return False
    return _fmt_yyyymmdd(d) not in A_SHARE_HOLIDAYS


def next_a_share_trading_day(trade_date: str, max_scan_days: int = 30) -> str:
    """返回给定日期之后的下一个 A 股交易日。"""
    d = _parse_yyyymmdd(trade_date)
    for _ in range(max_scan_days):
        d += timedelta(days=1)
        s = _fmt_yyyymmdd(d)
        if is_a_share_trading_day(s):
            return s
    raise RuntimeError(f"Cannot resolve next A-share trading day after {trade_date}")


def prev_a_share_trading_day(trade_date: str, max_scan_days: int = 30) -> str:
    """返回给定日期之前的上一个 A 股交易日。"""
    d = _parse_yyyymmdd(trade_date)
    for _ in range(max_scan_days):
        d -= timedelta(days=1)
        s = _fmt_yyyymmdd(d)
        if is_a_share_trading_day(s):
            return s
    raise RuntimeError(f"Cannot resolve previous A-share trading day before {trade_date}")


# =========================================================
# DataRepo —— 给 step0 / step5 使用的真实数据仓库接口
# =========================================================
class DataRepo:
    """
    统一访问你们本地数据仓库结构。

    仓库结构：
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
        """返回某个交易日快照目录 Path。"""
        year = str(trade_date)[:4]
        return self.warehouse_root / self.repo_name / self.raw_dir / year / str(trade_date)

    # ---------- 通用 CSV 读取 ----------
    @staticmethod
    def read_csv_if_exists(p: Path) -> pd.DataFrame:
        if not p.exists():
            return pd.DataFrame()
        for enc in ("utf-8", "utf-8-sig", "gbk"):
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

    def read_stk_auction(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "stk_auction.csv")

    def read_intraday_features(self, trade_date: str) -> pd.DataFrame:
        return self.read_csv_if_exists(self.snapshot_dir(trade_date) / "intraday_features.csv")

    def has_file(self, trade_date: str, filename: str) -> bool:
        return (self.snapshot_dir(trade_date) / filename).exists()

    def minute_dir(self, trade_date: str, freq: str = "1min") -> Path:
        return self.snapshot_dir(trade_date) / "minute" / freq

    # ---------- Step5 训练闭环需要：列出全部 snapshot 日期 ----------
    def list_snapshot_dates(self) -> list[str]:
        """
        返回所有已有 YYYYMMDD 快照目录，供 Step5 训练使用。
        注意：这是“已有数据日期”，不是完整交易日历。
        """
        root = self.warehouse_root / self.repo_name / self.raw_dir
        if not root.exists():
            return []

        dates: list[str] = []
        for year_dir in root.iterdir():
            if not year_dir.is_dir():
                continue
            for d in year_dir.iterdir():
                if d.is_dir() and len(d.name) == 8 and d.name.isdigit():
                    dates.append(d.name)

        return sorted(set(dates))

    # ---------- A 股交易日历能力 ----------
    def list_trade_dates(self, start: str = "20240101", end: str = "20261231") -> list[str]:
        """
        返回完整 A 股交易日历兜底表。
        不再把已有快照目录误当成完整交易日历。
        """
        s = _parse_yyyymmdd(start)
        e = _parse_yyyymmdd(end)
        out: list[str] = []
        cur = s
        while cur <= e:
            ds = _fmt_yyyymmdd(cur)
            if is_a_share_trading_day(ds):
                out.append(ds)
            cur += timedelta(days=1)
        return out

    def is_trade_date(self, trade_date: str) -> bool:
        return is_a_share_trading_day(str(trade_date))

    def prev_next_trade_date(self, trade_date: str) -> tuple[str, str]:
        """
        统一返回上一交易日 / 下一交易日。
        用于 writers、Step7、后续任何需要 verify_date 的模块。
        """
        return prev_a_share_trading_day(str(trade_date)), next_a_share_trading_day(str(trade_date))

    def next_trade_date(self, trade_date: str) -> str:
        return next_a_share_trading_day(str(trade_date))

    def prev_trade_date(self, trade_date: str) -> str:
        return prev_a_share_trading_day(str(trade_date))


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
class IntradayCfg:
    enabled: bool = True
    require_intraday_features: bool = False
    require_stk_auction: bool = False
    missing_policy: str = "neutral"
    quality_weights: Dict[str, float] = field(default_factory=dict)
    strength_plus: Dict[str, float] = field(default_factory=dict)
    final_score: Dict[str, float] = field(default_factory=dict)
    hard_filters: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Settings:
    version: str = "0.1"
    timezone: str = "Asia/Shanghai"

    data_repo: DataRepoCfg = field(default_factory=DataRepoCfg)
    io: IOCfg = field(default_factory=IOCfg)
    emotion_gate: EmotionGateCfg = field(default_factory=EmotionGateCfg)
    intraday: IntradayCfg = field(default_factory=IntradayCfg)

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
        today = datetime.now().strftime("%Y%m%d")
        if is_a_share_trading_day(today):
            return today
        return prev_a_share_trading_day(today)


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
    data_repo_cfg = DataRepoCfg(
        warehouse_root=str(dr.get("warehouse_root", DataRepoCfg.warehouse_root)),
        repo_name=str(dr.get("repo_name", DataRepoCfg.repo_name)),
        raw_dir=str(dr.get("raw_dir", DataRepoCfg.raw_dir)),
    )

    # 重要：重新生成 DataRepo 实例
    s.data_repo = DataRepo(
        warehouse_root=data_repo_cfg.warehouse_root,
        repo_name=data_repo_cfg.repo_name,
        raw_dir=data_repo_cfg.raw_dir,
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
        min_limit_up_cnt=int(eg.get("min_limit_up_cnt", s.emotion_gate.min_limit_up_cnt)),
        max_broken_rate=float(eg.get("max_broken_rate", s.emotion_gate.max_broken_rate)),
        min_max连板高度=int(eg.get("min_max连板高度", s.emotion_gate.min_max连板高度)),
    )

    # -------- intraday --------
    intraday_raw = raw.get("intraday", {}) or {}
    s.intraday = IntradayCfg(
        enabled=bool(intraday_raw.get("enabled", s.intraday.enabled)),
        require_intraday_features=bool(
            intraday_raw.get("require_intraday_features", s.intraday.require_intraday_features)
        ),
        require_stk_auction=bool(intraday_raw.get("require_stk_auction", s.intraday.require_stk_auction)),
        missing_policy=str(intraday_raw.get("missing_policy", s.intraday.missing_policy)),
        quality_weights=dict(intraday_raw.get("quality_weights", {}) or {}),
        strength_plus=dict(intraday_raw.get("strength_plus", {}) or {}),
        final_score=dict(intraday_raw.get("final_score", {}) or {}),
        hard_filters=dict(intraday_raw.get("hard_filters", {}) or {}),
        defaults=dict(intraday_raw.get("defaults", {}) or {}),
        report=dict(intraday_raw.get("report", {}) or {}),
    )

    # Preserve raw config blocks that downstream steps already probe with getattr().
    for block in ("ml", "training", "step6", "theme", "scores", "risk", "output"):
        if block in raw and isinstance(raw.get(block), dict):
            setattr(s, block, _dict_to_namespace(raw.get(block) or {}))

    return s


def _dict_to_namespace(obj: Dict[str, Any]) -> SimpleNamespace:
    converted = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            converted[k] = _dict_to_namespace(v)
        else:
            converted[k] = v
    return SimpleNamespace(**converted)
