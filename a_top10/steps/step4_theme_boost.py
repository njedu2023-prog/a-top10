# -*- coding: utf-8 -*-
"""
Step4 : Theme Boost — Top10 V3

定位：
- Step4 不再只是“题材加成凑一个分”，而是 V3 的题材热度特征聚合器。
- 负责把候选样本补齐为可下游消费的题材字段契约。
- 明确输出：
    1) 题材主分：ThemeBoost
    2) 题材层质量辅助字段：
       - theme_main
       - theme_count
       - theme_heat
       - board_rank
       - lhb_flag
       - lhb_net_buy
       - north_money_flag
       - theme_feature_count
       - theme_missing_fields
       - theme_quality_flag

V3 缺失策略：
- theme_heat 缺失：保留为空，不伪造 0
- board_rank 缺失：保留为空，不伪造 0
- lhb_flag 缺失：可置 0
- lhb_net_buy 缺失：保留为空，不伪造 0
- north_money_flag 缺失：可置 0

兼容要求：
- 保持 run_step4(s, ctx) -> ctx
- 保持 ctx["theme_df"] 输出
- 保持调试文件与 learning 落库
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from a_top10.config import Settings


# =========================================================
# Config
# =========================================================

INDUSTRY_COL_CANDIDATES = [
    "theme_main",
    "industry",
    "industry_name",
    "industryName",
    "申万行业",
    "申万一级行业",
    "一级行业",
    "行业",
    "行业名称",
    "板块",
    "板块名称",
    "board",
]

CODE_COL_CANDIDATES = [
    "ts_code",
    "code",
    "TS_CODE",
    "代码",
    "股票代码",
    "证券代码",
    "symbol",
    "Symbol",
    "ticker",
    "sec_code",
]

NAME_COL_CANDIDATES = [
    "name",
    "NAME",
    "名称",
    "股票名称",
    "证券名称",
    "ts_name",
    "TS_NAME",
    "股票简称",
    "证券简称",
]

HOT_BOARDS_INDUSTRY_COLS = [
    "industry",
    "industry_name",
    "行业",
    "板块",
    "板块名称",
    "行业名称",
]

HOT_BOARDS_RANK_COLS = [
    "rank",
    "Rank",
    "排名",
    "hot_rank",
    "热度排名",
]

HOT_BOARDS_HEAT_COLS = [
    "heat",
    "Heat",
    "score",
    "hot",
    "热度",
    "热度值",
    "板块热度",
]

TOP_LIST_CODE_COLS = [
    "ts_code",
    "code",
    "TS_CODE",
    "代码",
    "股票代码",
    "证券代码",
    "symbol",
    "Symbol",
    "ticker",
    "sec_code",
]

TOP_LIST_NET_COLS = [
    "net_amount",
    "net_amt",
    "net_buy",
    "net",
    "净买额",
    "净买入",
]

HSGT_CODE_COLS = [
    "ts_code",
    "code",
    "TS_CODE",
    "代码",
    "股票代码",
    "证券代码",
    "symbol",
    "Symbol",
    "ticker",
    "sec_code",
]

HSGT_NET_COLS = [
    "net_amount",
    "net_amt",
    "net_buy",
    "net",
    "north_net_buy",
    "buy_amount",
    "净买额",
    "净买入",
]

STEP4_LEARNING_REL_PATH = Path("outputs") / "learning" / "step4_theme.csv"
DEFAULT_RANK_DECAY_K = 0.18
DEFAULT_TOPK_INDUSTRY = 40
DEFAULT_DRAGON_BONUS = 0.08
DEFAULT_NORTH_BONUS = 0.05
DEFAULT_THEMECOUNT_MAX = 5
DEFAULT_FUZZY_MAX_ROWS = 300


# =========================================================
# Basic utils
# =========================================================

_RE_FIRST_INT = re.compile(r"(\d+)")
_RE_CODE6 = re.compile(r"(\d{6})")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_str(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "<na>", "none"):
        return ""
    return s


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        key = str(c).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _as_str_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    s = df[col].astype("object")
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    s = s.map(lambda v: pd.NA if isinstance(v, str) and v == "" else v)
    return s


def _to_float_nullable(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _clip01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.clip(0.0, 1.0)


def _norm_code(code: Any) -> Tuple[str, str]:
    s = _safe_str(code).upper()
    if not s:
        return "", ""
    m = _RE_CODE6.search(s)
    code6 = m.group(1) if m else s.split(".")[0]
    return s, code6


def _norm_industry_key(x: Any) -> str:
    s = _safe_str(x)
    if not s:
        return ""
    s2 = re.sub(r"[\s\u3000]+", "", s)
    for suf in ("行业", "板块", "概念"):
        if s2.endswith(suf) and len(s2) > len(suf):
            s2 = s2[: -len(suf)]
    return s2.lower()


def _parse_rank_int(x: Any, default: int = 9999) -> int:
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        v = int(x)
        return v if v > 0 else default
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return default
        v = int(x)
        return v if v > 0 else default

    s = _safe_str(x)
    if not s:
        return default

    m = _RE_FIRST_INT.search(s)
    if not m:
        return default
    try:
        v = int(m.group(1))
        return v if v > 0 else default
    except Exception:
        return default


def _read_csv_guess(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception:
            continue
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame()


def _ctx_get_df(ctx: Dict[str, Any], keys: Sequence[str]) -> pd.DataFrame:
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return pd.DataFrame()


def _ctx_get_path(ctx: Dict[str, Any], keys: Sequence[str]) -> Optional[Path]:
    for k in keys:
        v = ctx.get(k)
        if v is None:
            continue
        try:
            return Path(v)
        except Exception:
            continue
    return None


def _resolve_trade_date(ctx: Dict[str, Any]) -> str:
    for k in ("trade_date", "TRADE_DATE", "date", "run_date"):
        v = ctx.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return os.getenv("TRADE_DATE", "").strip() or "unknown"


def _ensure_outputs_dir(ctx: Dict[str, Any], s: Optional[Settings]) -> Path:
    p = _ctx_get_path(ctx, ["outputs_dir", "output_dir", "out_dir"])
    if p is not None:
        p.mkdir(parents=True, exist_ok=True)
        return p

    if s is not None:
        try:
            io = getattr(s, "io", None)
            if io is not None and hasattr(io, "outputs_dir") and getattr(io, "outputs_dir"):
                p2 = Path(getattr(io, "outputs_dir"))
                p2.mkdir(parents=True, exist_ok=True)
                return p2
        except Exception:
            pass

    p3 = Path("outputs")
    p3.mkdir(parents=True, exist_ok=True)
    return p3


def _ensure_learning_dir(ctx: Dict[str, Any], s: Optional[Settings]) -> Path:
    p = _ensure_outputs_dir(ctx, s) / "learning"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_weights_value(s: Optional[Settings], key: str, default: Any) -> Any:
    if s is None:
        return default
    w = getattr(s, "weights", None)
    if w is None:
        return default
    try:
        if isinstance(w, dict):
            return w.get(key, default)
        return getattr(w, key, default)
    except Exception:
        return default


def _get_rank_decay_k(s: Optional[Settings]) -> float:
    try:
        return float(_get_weights_value(s, "rank_decay_k", DEFAULT_RANK_DECAY_K))
    except Exception:
        return float(DEFAULT_RANK_DECAY_K)


def _get_topk_industry(s: Optional[Settings]) -> int:
    try:
        return int(_get_weights_value(s, "topk_industry", DEFAULT_TOPK_INDUSTRY))
    except Exception:
        return int(DEFAULT_TOPK_INDUSTRY)


def _get_dragon_bonus(s: Optional[Settings]) -> float:
    try:
        return float(_get_weights_value(s, "dragon_bonus", DEFAULT_DRAGON_BONUS))
    except Exception:
        return float(DEFAULT_DRAGON_BONUS)


def _get_north_bonus(s: Optional[Settings]) -> float:
    try:
        return float(_get_weights_value(s, "north_bonus", DEFAULT_NORTH_BONUS))
    except Exception:
        return float(DEFAULT_NORTH_BONUS)


def _get_fuzzy_max_rows(s: Optional[Settings]) -> int:
    try:
        return int(_get_weights_value(s, "fuzzy_max_rows", DEFAULT_FUZZY_MAX_ROWS))
    except Exception:
        return int(DEFAULT_FUZZY_MAX_ROWS)


# =========================================================
# Snapshot / candidate resolve
# =========================================================

def _candidate_snapshot_dirs(trade_date: str) -> Sequence[Path]:
    y = trade_date[:4] if trade_date and trade_date[:4].isdigit() else ""
    dirs: List[Path] = []
    if y:
        dirs.append(Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / y / trade_date)
    dirs.extend(
        [
            Path("_warehouse") / "a-share-top3-data" / "data" / "raw" / trade_date,
            Path("data_repo") / "snapshots" / trade_date,
            Path("snapshots") / trade_date,
        ]
    )
    return dirs


def _locate_snapshot_dir(ctx: Dict[str, Any], trade_date: str) -> Optional[Path]:
    p = _ctx_get_path(ctx, ["snapshot_dir", "snap_dir", "snapshot_path"])
    if p is not None and p.exists():
        return p
    for d in _candidate_snapshot_dirs(trade_date):
        if d.exists():
            return d
    return None


def _load_snapshot_tables(snapshot_dir: Optional[Path]) -> Dict[str, pd.DataFrame]:
    if snapshot_dir is None:
        return {
            "hot_boards": pd.DataFrame(),
            "stock_basic": pd.DataFrame(),
            "top_list": pd.DataFrame(),
            "moneyflow_hsgt": pd.DataFrame(),
        }

    sd = Path(snapshot_dir)
    return {
        "hot_boards": _read_csv_guess(sd / "hot_boards.csv"),
        "stock_basic": _read_csv_guess(sd / "stock_basic.csv"),
        "top_list": _read_csv_guess(sd / "top_list.csv"),
        "moneyflow_hsgt": _read_csv_guess(sd / "moneyflow_hsgt.csv"),
    }


def _coerce_candidate_df(ctx: Dict[str, Any]) -> pd.DataFrame:
    cand_df = _ctx_get_df(
        ctx,
        [
            "step3_df",
            "step3_strength_df",
            "step3_out",
            "candidates",
            "candidate_df",
            "df",
        ],
    )
    if cand_df.empty:
        v = ctx.get("step3", None)
        if isinstance(v, Mapping):
            maybe = v.get("df")
            if isinstance(maybe, pd.DataFrame):
                cand_df = maybe
    return cand_df.copy() if isinstance(cand_df, pd.DataFrame) else pd.DataFrame()


def _ensure_identity(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    code_col = _first_existing_col(out, CODE_COL_CANDIDATES) or "ts_code"
    out["ts_code"] = _as_str_series(out, code_col)

    name_col = _first_existing_col(out, NAME_COL_CANDIDATES)
    out["name"] = _as_str_series(out, name_col) if name_col else ""

    td_col = _first_existing_col(out, ["trade_date", "TRADE_DATE", "日期"])
    if td_col:
        out["trade_date"] = out[td_col].astype(str)
    else:
        out["trade_date"] = trade_date

    return out


# =========================================================
# Step5 passthrough compatibility
# =========================================================

_STEP5_FEATURE_ALIASES: Dict[str, Sequence[str]] = {
    "StrengthScore": ["StrengthScore", "strengthscore", "strength_score", "Strength", "strength"],
    "ThemeBoost": ["ThemeBoost", "themeboost", "theme_boost", "题材加成", "题材", "Theme"],
    "seal_amount": ["seal_amount", "sealamount", "seal_amt", "seal", "封单额", "封单金额", "封单"],
    "open_times": ["open_times", "opentimes", "open_time", "open_count", "openings", "开板次数", "炸板次数"],
    "turnover_rate": ["turnover_rate", "turnoverrate", "turn_rate", "turnover", "换手率"],
}


def _ensure_step5_features_passthrough(df: pd.DataFrame, dbg: Dict[str, Any]) -> pd.DataFrame:
    """
    V3 收口要求：
    - 只做别名归并，不把缺失伪造为 0
    - Step3 的关键字段允许原样透传空值
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    lower_map = {str(c).strip().lower(): c for c in out.columns}

    missing: List[str] = []
    mapped: Dict[str, str] = {}

    for canon, aliases in _STEP5_FEATURE_ALIASES.items():
        if canon in out.columns:
            continue

        found_col = None
        if canon.lower() in lower_map:
            found_col = lower_map[canon.lower()]
        else:
            for a in aliases:
                ak = str(a).strip().lower()
                if ak in lower_map:
                    found_col = lower_map[ak]
                    break

        if found_col:
            out[canon] = out[found_col]
            mapped[canon] = found_col
        else:
            out[canon] = np.nan
            missing.append(canon)

    # 兼容旧 writers / 旧消费口径
    if "题材加成" not in out.columns and "ThemeBoost" in out.columns:
        out["题材加成"] = out["ThemeBoost"]
    if "board" not in out.columns and "板块" in out.columns:
        out["board"] = out["板块"]

    dbg["step5_feature_passthrough"] = {"mapped": mapped, "missing": missing}

    return out


# =========================================================
# Industry / board heat mapping
# =========================================================

def _build_board_heat_map(
    hot_boards: pd.DataFrame,
    rank_decay_k: float,
    topk_industry: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    dbg: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "industry_col": "",
        "rank_col": "",
        "heat_col": "",
        "size": 0,
        "top5": [],
    }

    if hot_boards is None or hot_boards.empty:
        dbg["reason"] = "hot_boards empty"
        return {}, dbg

    ind_col = _first_existing_col(hot_boards, HOT_BOARDS_INDUSTRY_COLS)
    rank_col = _first_existing_col(hot_boards, HOT_BOARDS_RANK_COLS)
    heat_col = _first_existing_col(hot_boards, HOT_BOARDS_HEAT_COLS)

    dbg["industry_col"] = ind_col or ""
    dbg["rank_col"] = rank_col or ""
    dbg["heat_col"] = heat_col or ""

    if not ind_col:
        dbg["reason"] = "hot_boards missing industry col"
        return {}, dbg

    df = hot_boards.copy()
    df[ind_col] = df[ind_col].astype(str).map(_safe_str)
    df = df[df[ind_col] != ""].copy()
    if df.empty:
        dbg["reason"] = "hot_boards industry all blank"
        return {}, dbg

    mp: Dict[str, Dict[str, Any]] = {}

    # 优先用显式热度列；若没有，再回退到 rank
    if heat_col:
        x = df[[ind_col, heat_col]].copy()
        x[heat_col] = pd.to_numeric(x[heat_col], errors="coerce")
        x = x.dropna(subset=[heat_col]).copy()
        if x.empty:
            dbg["reason"] = "hot_boards heat parse failed"
            return {}, dbg

        x = x.sort_values(heat_col, ascending=False).drop_duplicates(subset=[ind_col], keep="first")
        if topk_industry > 0:
            x = x.head(int(topk_industry))

        vals = x[heat_col].astype("float64").to_numpy()
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if abs(vmax - vmin) > 1e-12:
            heat_norm = (vals - vmin) / (vmax - vmin)
        else:
            heat_norm = np.ones_like(vals, dtype=float)

        x = x.reset_index(drop=True)
        x["board_rank_tmp"] = np.arange(1, len(x) + 1)

        for _, row in x.iterrows():
            raw_ind = _safe_str(row[ind_col])
            nk = _norm_industry_key(raw_ind)
            if not nk:
                continue
            mp[nk] = {
                "industry_raw": raw_ind,
                "theme_heat": float(np.clip(heat_norm[int(row.name)], 0.0, 1.0)),
                "board_rank": int(row["board_rank_tmp"]),
            }

    elif rank_col:
        x = df[[ind_col, rank_col]].copy()
        x[rank_col] = x[rank_col].map(lambda v: _parse_rank_int(v, default=9999))
        x = x[x[rank_col] < 9999].copy()
        if x.empty:
            dbg["reason"] = "hot_boards rank parse failed"
            return {}, dbg

        x = x.sort_values(rank_col, ascending=True).drop_duplicates(subset=[ind_col], keep="first")
        if topk_industry > 0:
            x = x.head(int(topk_industry))

        ranks = x[rank_col].astype(int).to_numpy()
        scores = np.exp(-float(rank_decay_k) * (ranks - 1.0))
        smin = float(np.min(scores))
        smax = float(np.max(scores))
        if abs(smax - smin) > 1e-12:
            scores = (scores - smin) / (smax - smin)
        scores = np.clip(scores, 0.0, 1.0)

        for idx, (_, row) in enumerate(x.iterrows()):
            raw_ind = _safe_str(row[ind_col])
            nk = _norm_industry_key(raw_ind)
            if not nk:
                continue
            mp[nk] = {
                "industry_raw": raw_ind,
                "theme_heat": float(scores[idx]),
                "board_rank": int(row[rank_col]),
            }
    else:
        dbg["reason"] = "hot_boards missing both rank and heat col"
        return {}, dbg

    dbg["ok"] = True
    dbg["size"] = len(mp)
    dbg["top5"] = [
        {"industry": v["industry_raw"], "theme_heat": v["theme_heat"], "board_rank": v["board_rank"]}
        for _, v in sorted(mp.items(), key=lambda kv: kv[1]["theme_heat"], reverse=True)[:5]
    ]
    return mp, dbg


def _enrich_theme_main_from_stock_basic(
    out: pd.DataFrame,
    stock_basic: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, float, float]:
    ind_col_before = _first_existing_col(out, INDUSTRY_COL_CANDIDATES)
    before_ratio = (
        float((_as_str_series(out, ind_col_before).fillna("") != "").mean())
        if ind_col_before else 0.0
    )

    need_enrich = (not ind_col_before) or (before_ratio < 0.1)
    if not need_enrich or stock_basic is None or stock_basic.empty:
        if ind_col_before and ind_col_before in out.columns:
            out["theme_main"] = _as_str_series(out, ind_col_before)
        else:
            out["theme_main"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
        after_ratio = float((out["theme_main"].fillna("") != "").mean()) if len(out) else 0.0
        return out, ind_col_before or "theme_main", before_ratio, after_ratio

    sb_code_col = _first_existing_col(stock_basic, CODE_COL_CANDIDATES)
    sb_ind_col = _first_existing_col(stock_basic, INDUSTRY_COL_CANDIDATES)
    if not sb_code_col or not sb_ind_col:
        out["theme_main"] = _as_str_series(out, ind_col_before) if ind_col_before else pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
        after_ratio = float((out["theme_main"].fillna("") != "").mean()) if len(out) else 0.0
        return out, ind_col_before or "theme_main", before_ratio, after_ratio

    tmp = stock_basic[[sb_code_col, sb_ind_col]].copy()
    tmp["_code6"] = tmp[sb_code_col].map(lambda x: _norm_code(x)[1])
    tmp = tmp[tmp["_code6"].astype(str).map(_safe_str) != ""].drop_duplicates(subset=["_code6"], keep="first")

    out2 = out.copy()
    out2["_code6"] = out2["ts_code"].map(lambda x: _norm_code(x)[1])
    out2 = out2.merge(tmp[["_code6", sb_ind_col]], on="_code6", how="left")

    if ind_col_before and ind_col_before in out2.columns:
        s0 = _as_str_series(out2, ind_col_before)
    else:
        s0 = pd.Series([pd.NA] * len(out2), index=out2.index, dtype="object")

    sb_series = _as_str_series(out2, sb_ind_col)
    theme_main = s0.where(s0.notna() & (s0 != ""), sb_series)
    out2["theme_main"] = theme_main

    after_ratio = float((out2["theme_main"].fillna("") != "").mean()) if len(out2) else 0.0
    return out2, "theme_main", before_ratio, after_ratio


# =========================================================
# Theme feature extraction
# =========================================================

def _infer_theme_count(out: pd.DataFrame) -> pd.Series:
    """
    优先读取已有字段；若没有，则从 theme_main / board / 概念串粗略估计。
    """
    explicit_col = _first_existing_col(
        out,
        [
            "theme_count",
            "theme_num",
            "board_count",
            "concept_count",
            "题材数",
            "题材数量",
            "概念数",
        ],
    )
    if explicit_col:
        s = pd.to_numeric(out[explicit_col], errors="coerce").astype("float64")
        return s.replace([np.inf, -np.inf], np.nan)

    text_col = _first_existing_col(
        out,
        [
            "concepts",
            "concept",
            "board",
            "板块",
            "概念",
            "题材",
            "theme_main",
        ],
    )
    if not text_col:
        return pd.Series([np.nan] * len(out), index=out.index, dtype="float64")

    def _count_parts(x: Any) -> float:
        s = _safe_str(x)
        if not s:
            return np.nan
        parts = re.split(r"[;；,，/\|\s]+", s)
        parts = [p for p in parts if _safe_str(p)]
        if not parts:
            return np.nan
        return float(len(set(parts)))

    return out[text_col].map(_count_parts).astype("float64")


def _build_lhb_maps(top_list: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, float]]:
    if top_list is None or top_list.empty:
        return {}, {}

    code_col = _first_existing_col(top_list, TOP_LIST_CODE_COLS)
    if not code_col:
        return {}, {}

    net_col = _first_existing_col(top_list, TOP_LIST_NET_COLS)

    flag_map: Dict[str, int] = {}
    net_map: Dict[str, float] = {}

    for _, row in top_list.iterrows():
        ts, c6 = _norm_code(row.get(code_col))
        if not ts and not c6:
            continue

        key_list = [k for k in (ts, c6) if k]
        net_v = np.nan
        if net_col:
            try:
                net_v = float(pd.to_numeric(row.get(net_col), errors="coerce"))
            except Exception:
                net_v = np.nan

        for k in key_list:
            flag_map[k] = 1
            if not np.isnan(net_v):
                old = net_map.get(k, 0.0)
                net_map[k] = float(old) + float(net_v)

    return flag_map, net_map


def _build_north_maps(hsgt: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, float]]:
    if hsgt is None or hsgt.empty:
        return {}, {}

    code_col = _first_existing_col(hsgt, HSGT_CODE_COLS)
    if not code_col:
        return {}, {}

    net_col = _first_existing_col(hsgt, HSGT_NET_COLS)

    flag_map: Dict[str, int] = {}
    net_map: Dict[str, float] = {}

    for _, row in hsgt.iterrows():
        ts, c6 = _norm_code(row.get(code_col))
        if not ts and not c6:
            continue

        net_v = np.nan
        if net_col:
            try:
                net_v = float(pd.to_numeric(row.get(net_col), errors="coerce"))
            except Exception:
                net_v = np.nan

        keys = [k for k in (ts, c6) if k]
        positive_flag = 1 if (not np.isnan(net_v) and net_v > 0) else 0

        for k in keys:
            flag_map[k] = max(flag_map.get(k, 0), positive_flag)
            if not np.isnan(net_v):
                net_map[k] = float(net_map.get(k, 0.0)) + float(net_v)

    return flag_map, net_map


def _apply_board_heat_and_events(
    cand_df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    top_list: pd.DataFrame,
    hsgt: pd.DataFrame,
    board_heat_map: Dict[str, Dict[str, Any]],
    dragon_bonus: float,
    north_bonus: float,
    fuzzy_max_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dbg: Dict[str, Any] = {
        "candidate_rows": 0,
        "candidate_code_col": "",
        "theme_main_before_ratio": 0.0,
        "theme_main_after_ratio": 0.0,
        "matched_industry_count": 0,
        "matched_industry_exact": 0,
        "matched_industry_norm": 0,
        "matched_industry_fuzzy": 0,
        "lhb_hits": 0,
        "north_hits": 0,
        "reason": "",
    }

    if cand_df is None or cand_df.empty:
        dbg["reason"] = "candidate df empty"
        return pd.DataFrame(), dbg

    out = cand_df.copy()
    out = _ensure_identity(out, trade_date=_safe_str(_first_existing_col(out, ["trade_date"]) or ""))

    dbg["candidate_rows"] = int(len(out))
    dbg["candidate_code_col"] = "ts_code"

    out, theme_col, before_ratio, after_ratio = _enrich_theme_main_from_stock_basic(out, stock_basic)
    dbg["theme_main_before_ratio"] = float(before_ratio)
    dbg["theme_main_after_ratio"] = float(after_ratio)

    out["theme_main"] = _as_str_series(out, theme_col)

    theme_count = _infer_theme_count(out)
    out["theme_count"] = theme_count

    theme_heat = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")
    board_rank = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")

    exact_hits = 0
    norm_hits = 0
    fuzzy_hits = 0

    nk_to_payload = {k: v for k, v in (board_heat_map or {}).items()}
    nk_keys = list(nk_to_payload.keys())

    raw_theme = out["theme_main"].fillna("").astype(str)
    norm_theme = raw_theme.map(_norm_industry_key)

    # exact / normalized
    for idx in out.index:
        raw = raw_theme.at[idx]
        nk = norm_theme.at[idx]
        if not raw and not nk:
            continue

        payload = None
        if nk in nk_to_payload:
            payload = nk_to_payload[nk]
            norm_hits += 1
        elif _safe_str(raw) in [v["industry_raw"] for v in nk_to_payload.values()]:
            # 理论上一般不会走到这里，仍保留
            for v in nk_to_payload.values():
                if v["industry_raw"] == _safe_str(raw):
                    payload = v
                    exact_hits += 1
                    break

        if payload is not None:
            theme_heat.at[idx] = float(payload["theme_heat"])
            board_rank.at[idx] = float(payload["board_rank"])

    # fuzzy contains
    remain_mask = theme_heat.isna() & (norm_theme != "")
    remain_pos = np.where(remain_mask.to_numpy())[0]
    if len(remain_pos) > 0 and nk_keys:
        if fuzzy_max_rows and len(remain_pos) > int(fuzzy_max_rows):
            remain_pos = remain_pos[: int(fuzzy_max_rows)]
        for pos in remain_pos:
            nk = norm_theme.iat[pos]
            best = None
            for kk in nk_keys:
                if not kk:
                    continue
                if (kk in nk) or (nk in kk):
                    payload = nk_to_payload.get(kk)
                    if payload is None:
                        continue
                    if best is None or float(payload["theme_heat"]) > float(best["theme_heat"]):
                        best = payload
            if best is not None:
                theme_heat.iat[pos] = float(best["theme_heat"])
                board_rank.iat[pos] = float(best["board_rank"])
                fuzzy_hits += 1

    out["theme_heat"] = theme_heat
    out["board_rank"] = board_rank

    matched_total = int(out["theme_heat"].notna().sum())
    dbg["matched_industry_count"] = matched_total
    dbg["matched_industry_exact"] = int(exact_hits)
    dbg["matched_industry_norm"] = int(norm_hits)
    dbg["matched_industry_fuzzy"] = int(fuzzy_hits)

    # 龙虎榜 / 北向
    lhb_flag_map, lhb_net_map = _build_lhb_maps(top_list)
    north_flag_map, _north_net_map = _build_north_maps(hsgt)

    lhb_flag = []
    lhb_net_buy = []
    north_money_flag = []

    for _, row in out.iterrows():
        ts, c6 = _norm_code(row.get("ts_code"))
        keys = [k for k in (ts, c6) if k]

        lhb_hit = 0
        lhb_net = np.nan
        north_hit = 0

        for k in keys:
            if lhb_flag_map.get(k, 0) == 1:
                lhb_hit = 1
            if k in lhb_net_map:
                lhb_net = float(lhb_net_map[k]) if np.isnan(lhb_net) else float(lhb_net) + float(lhb_net_map[k])
            if north_flag_map.get(k, 0) == 1:
                north_hit = 1

        lhb_flag.append(float(lhb_hit))
        lhb_net_buy.append(lhb_net)
        north_money_flag.append(float(north_hit))

    out["lhb_flag"] = pd.Series(lhb_flag, index=out.index, dtype="float64")
    out["lhb_net_buy"] = pd.Series(lhb_net_buy, index=out.index, dtype="float64")
    out["north_money_flag"] = pd.Series(north_money_flag, index=out.index, dtype="float64")

    dbg["lhb_hits"] = int((out["lhb_flag"].fillna(0.0) > 0.5).sum())
    dbg["north_hits"] = int((out["north_money_flag"].fillna(0.0) > 0.5).sum())

    # ThemeBoost 计算：内部可兜底，但契约字段本身保留空值
    theme_heat_calc = out["theme_heat"].fillna(0.0)

    rank_score = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    if out["board_rank"].notna().any():
        valid_rank = out["board_rank"].dropna()
        max_rank = max(float(valid_rank.max()), 1.0)
        rank_score = (1.0 - ((out["board_rank"].fillna(max_rank) - 1.0) / max(max_rank - 1.0, 1.0))).clip(0.0, 1.0)

    theme_count_score = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    if out["theme_count"].notna().any():
        theme_count_score = (out["theme_count"].fillna(0.0) / float(DEFAULT_THEMECOUNT_MAX)).clip(0.0, 1.0)

    lhb_bonus = (out["lhb_flag"].fillna(0.0) > 0.5).astype("float64") * float(dragon_bonus)
    north_bonus_arr = (out["north_money_flag"].fillna(0.0) > 0.5).astype("float64") * float(north_bonus)

    boost = (
        0.70 * theme_heat_calc
        + 0.12 * rank_score
        + 0.08 * lhb_bonus
        + 0.05 * north_bonus_arr
        + 0.05 * theme_count_score
    ).clip(0.0, 1.0)

    out["ThemeBoost"] = boost.astype("float64")
    out["题材加成"] = out["ThemeBoost"]
    out["板块"] = out["theme_main"].fillna("").astype(str)

    # 质量辅助字段
    feature_cols = [
        "theme_main",
        "theme_count",
        "theme_heat",
        "board_rank",
        "lhb_flag",
        "lhb_net_buy",
        "north_money_flag",
    ]

    feature_count = pd.Series([0] * len(out), index=out.index, dtype="int64")
    for c in feature_cols:
        if c in out.columns:
            feature_count += out[c].notna().astype("int64")
    out["theme_feature_count"] = feature_count

    missing_fields: List[str] = []
    for idx in out.index:
        miss = []
        for c in ["theme_main", "theme_heat", "board_rank", "lhb_net_buy"]:
            if c not in out.columns or pd.isna(out.at[idx, c]) or (c == "theme_main" and _safe_str(out.at[idx, c]) == ""):
                miss.append(c)
        missing_fields.append("|".join(miss) if miss else "")
    out["theme_missing_fields"] = missing_fields

    has_main = out["theme_main"].fillna("").astype(str).map(lambda x: 1 if _safe_str(x) else 0)
    has_heat = out["theme_heat"].notna().astype(int)
    has_rank = out["board_rank"].notna().astype(int)
    has_enhance = (
        out["lhb_flag"].notna().astype(int)
        + out["lhb_net_buy"].notna().astype(int)
        + out["north_money_flag"].notna().astype(int)
    )

    q = pd.Series(["D"] * len(out), index=out.index, dtype="object")
    cond_c = (has_main == 1) | (has_enhance >= 1)
    cond_b = (has_main == 1) & ((has_heat + has_rank) >= 1)
    cond_a = (has_main == 1) & (has_heat == 1) & (has_rank == 1)

    q.loc[cond_c] = "C"
    q.loc[cond_b] = "B"
    q.loc[cond_a] = "A"
    out["theme_quality_flag"] = q

    # 调试细项
    out["_theme_heat_calc"] = theme_heat_calc
    out["_rank_score_calc"] = rank_score
    out["_theme_count_score"] = theme_count_score
    out["_lhb_bonus_calc"] = lhb_bonus
    out["_north_bonus_calc"] = north_bonus_arr

    if not board_heat_map:
        dbg["reason"] = "board heat map empty"
    elif matched_total == 0 and dbg["lhb_hits"] > 0:
        dbg["reason"] = "industry heat not applied; only lhb bonus applied"
    elif matched_total == 0 and dbg["lhb_hits"] == 0 and dbg["north_hits"] == 0:
        dbg["reason"] = "industry heat / lhb / north all not applied"
    else:
        dbg["reason"] = "theme boost applied"

    return out, dbg


# =========================================================
# Learning write
# =========================================================

def _build_step4_learning_frame(out_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if out_df is None or out_df.empty:
        return pd.DataFrame(
            columns=[
                "run_time_utc",
                "trade_date",
                "ts_code",
                "name",
                "theme_main",
                "theme_heat",
                "board_rank",
                "lhb_flag",
                "lhb_net_buy",
                "north_money_flag",
                "ThemeBoost",
                "theme_feature_count",
                "theme_quality_flag",
            ]
        )

    df = out_df.copy()
    run_time_utc = _utc_now_iso()

    out = pd.DataFrame(
        {
            "run_time_utc": run_time_utc,
            "trade_date": _safe_str(trade_date),
            "ts_code": df["ts_code"].astype(str).map(_safe_str),
            "name": df["name"].astype(str).map(_safe_str) if "name" in df.columns else "",
            "theme_main": df["theme_main"].astype(str).map(_safe_str) if "theme_main" in df.columns else "",
            "theme_heat": pd.to_numeric(df.get("theme_heat"), errors="coerce"),
            "board_rank": pd.to_numeric(df.get("board_rank"), errors="coerce"),
            "lhb_flag": pd.to_numeric(df.get("lhb_flag"), errors="coerce"),
            "lhb_net_buy": pd.to_numeric(df.get("lhb_net_buy"), errors="coerce"),
            "north_money_flag": pd.to_numeric(df.get("north_money_flag"), errors="coerce"),
            "ThemeBoost": pd.to_numeric(df.get("ThemeBoost"), errors="coerce"),
            "theme_feature_count": pd.to_numeric(df.get("theme_feature_count"), errors="coerce"),
            "theme_quality_flag": df["theme_quality_flag"].astype(str).map(_safe_str) if "theme_quality_flag" in df.columns else "",
        }
    )
    out = out[out["ts_code"] != ""].copy()
    return out


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def _upsert_step4_learning(learn_df: pd.DataFrame, learning_path: Path) -> Dict[str, Any]:
    dbg: Dict[str, Any] = {"ok": False, "path": str(learning_path), "rows_in": int(len(learn_df))}

    if learn_df is None or learn_df.empty:
        dbg["reason"] = "learn_df empty"
        return dbg

    old = _read_csv_guess(learning_path)
    dbg["rows_old"] = int(len(old)) if isinstance(old, pd.DataFrame) else 0

    merged = learn_df.copy() if old.empty else pd.concat([old, learn_df], ignore_index=True, sort=False)

    required_cols = [
        "run_time_utc",
        "trade_date",
        "ts_code",
        "name",
        "theme_main",
        "theme_heat",
        "board_rank",
        "lhb_flag",
        "lhb_net_buy",
        "north_money_flag",
        "ThemeBoost",
        "theme_feature_count",
        "theme_quality_flag",
    ]
    for c in required_cols:
        if c not in merged.columns:
            merged[c] = ""

    merged["trade_date"] = merged["trade_date"].astype(str).map(_safe_str)
    merged["ts_code"] = merged["ts_code"].astype(str).map(_safe_str)
    merged["run_time_utc"] = merged["run_time_utc"].astype(str).map(_safe_str)
    merged = merged.sort_values(["trade_date", "ts_code", "run_time_utc"], ascending=[True, True, True])
    merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last").reset_index(drop=True)

    _atomic_write_csv(merged, learning_path)

    dbg["ok"] = True
    dbg["rows_new_total"] = int(len(merged))
    return dbg


# =========================================================
# Output selection / debug
# =========================================================

def _formal_output_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "ts_code",
        "name",
        "trade_date",
        "theme_main",
        "theme_count",
        "theme_heat",
        "board_rank",
        "lhb_flag",
        "lhb_net_buy",
        "north_money_flag",
        "ThemeBoost",
        "题材加成",
        "板块",
        "theme_feature_count",
        "theme_missing_fields",
        "theme_quality_flag",
        "StrengthScore",
        "turnover_rate",
        "seal_amount",
        "open_times",
        "_theme_heat_calc",
        "_rank_score_calc",
        "_theme_count_score",
        "_lhb_bonus_calc",
        "_north_bonus_calc",
    ]
    return [c for c in cols if c in df.columns]


# =========================================================
# Entry
# =========================================================

def run_step4(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入：
      ctx 中应有候选 df（通常 step3 输出）
    输出：
      ctx["theme_df"]：带 ThemeBoost / 题材层质量字段的 df
      ctx["debug"]["step4_theme"]：调试信息
      并尝试落盘：
        outputs/debug_step4_theme_YYYYMMDD.json
        outputs/learning/step4_theme.csv
    """
    ctx = ctx or {}
    ctx.setdefault("debug", {})
    dbg_all: Dict[str, Any] = {}

    trade_date = _resolve_trade_date(ctx)
    cand_df = _coerce_candidate_df(ctx)

    snapshot_dir = _locate_snapshot_dir(ctx, trade_date)
    snap = _load_snapshot_tables(snapshot_dir)

    hot_boards = _ctx_get_df(ctx, ["hot_boards", "hot_boards_df"])
    stock_basic = _ctx_get_df(ctx, ["stock_basic", "stock_basic_df"])
    top_list = _ctx_get_df(ctx, ["top_list", "top_list_df", "龙虎榜", "longhu"])
    moneyflow_hsgt = _ctx_get_df(ctx, ["moneyflow_hsgt", "moneyflow_hsgt_df", "north_money", "hsgt_df"])

    if hot_boards.empty:
        hot_boards = snap["hot_boards"]
    if stock_basic.empty:
        stock_basic = snap["stock_basic"]
    if top_list.empty:
        top_list = snap["top_list"]
    if moneyflow_hsgt.empty:
        moneyflow_hsgt = snap["moneyflow_hsgt"]

    rank_decay_k = _get_rank_decay_k(s)
    topk_industry = _get_topk_industry(s)
    dragon_bonus = _get_dragon_bonus(s)
    north_bonus = _get_north_bonus(s)
    fuzzy_max_rows = _get_fuzzy_max_rows(s)

    board_heat_map, dbg_map = _build_board_heat_map(
        hot_boards=hot_boards,
        rank_decay_k=rank_decay_k,
        topk_industry=topk_industry,
    )

    out_df, dbg_apply = _apply_board_heat_and_events(
        cand_df=cand_df,
        stock_basic=stock_basic,
        top_list=top_list,
        hsgt=moneyflow_hsgt,
        board_heat_map=board_heat_map,
        dragon_bonus=dragon_bonus,
        north_bonus=north_bonus,
        fuzzy_max_rows=fuzzy_max_rows,
    )

    out_df = _ensure_step5_features_passthrough(out_df, dbg_all)

    # 正式输出列集优先
    formal_cols = _formal_output_columns(out_df)
    out_df = out_df[formal_cols + [c for c in out_df.columns if c not in formal_cols]].copy()

    dbg_all["trade_date"] = trade_date
    dbg_all["snapshot_dir"] = str(snapshot_dir) if snapshot_dir is not None else ""
    dbg_all["params"] = {
        "rank_decay_k": float(rank_decay_k),
        "topk_industry": int(topk_industry),
        "dragon_bonus": float(dragon_bonus),
        "north_bonus": float(north_bonus),
        "fuzzy_max_rows": int(fuzzy_max_rows),
    }
    dbg_all["source_rows"] = {
        "candidate": int(len(cand_df)) if isinstance(cand_df, pd.DataFrame) else 0,
        "hot_boards": int(len(hot_boards)) if isinstance(hot_boards, pd.DataFrame) else 0,
        "stock_basic": int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0,
        "top_list": int(len(top_list)) if isinstance(top_list, pd.DataFrame) else 0,
        "moneyflow_hsgt": int(len(moneyflow_hsgt)) if isinstance(moneyflow_hsgt, pd.DataFrame) else 0,
    }
    dbg_all["industry_score_map"] = dbg_map
    dbg_all["apply"] = dbg_apply

    if out_df.empty:
        dbg_all["diagnosis"] = "theme_df empty"
    elif int(dbg_apply.get("matched_industry_count", 0)) == 0 and int(dbg_apply.get("lhb_hits", 0)) == 0 and int(dbg_apply.get("north_hits", 0)) == 0:
        dbg_all["diagnosis"] = "industry heat / lhb / north all not applied"
    elif int(dbg_apply.get("matched_industry_count", 0)) == 0 and (
        int(dbg_apply.get("lhb_hits", 0)) > 0 or int(dbg_apply.get("north_hits", 0)) > 0
    ):
        dbg_all["diagnosis"] = "industry heat not applied; event bonus applied"
    else:
        dbg_all["diagnosis"] = "theme boost applied"

    # 质量概览
    if not out_df.empty:
        dbg_all["quality_distribution"] = (
            out_df["theme_quality_flag"].value_counts(dropna=False).to_dict()
            if "theme_quality_flag" in out_df.columns else {}
        )
        dbg_all["missing_rate"] = {
            "theme_main": float(1.0 - (out_df["theme_main"].fillna("").astype(str).map(lambda x: 1 if _safe_str(x) else 0).mean())) if "theme_main" in out_df.columns else 1.0,
            "theme_heat": float(1.0 - out_df["theme_heat"].notna().mean()) if "theme_heat" in out_df.columns else 1.0,
            "board_rank": float(1.0 - out_df["board_rank"].notna().mean()) if "board_rank" in out_df.columns else 1.0,
            "lhb_net_buy": float(1.0 - out_df["lhb_net_buy"].notna().mean()) if "lhb_net_buy" in out_df.columns else 1.0,
            "ThemeBoost": float(1.0 - out_df["ThemeBoost"].notna().mean()) if "ThemeBoost" in out_df.columns else 1.0,
        }

    ctx["theme_df"] = out_df
    ctx["debug"]["step4_theme"] = dbg_all

    # 调试文件
    try:
        out_dir = _ensure_outputs_dir(ctx, s)
        dbg_path = out_dir / f"debug_step4_theme_{trade_date}.json"
        dbg_path.write_text(json.dumps(dbg_all, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx["debug"]["step4_theme"]["debug_file"] = str(dbg_path)
    except Exception as e:
        ctx["debug"]["step4_theme"]["debug_file"] = f"write_failed:{type(e).__name__}"

    # learning 落库
    try:
        learning_dir = _ensure_learning_dir(ctx, s)
        learning_path = learning_dir / "step4_theme.csv"
        learn_df = _build_step4_learning_frame(out_df=out_df, trade_date=trade_date)
        dbg_write = _upsert_step4_learning(learn_df, learning_path)
        ctx["debug"]["step4_theme"]["learning_write"] = dbg_write
        ctx["debug"]["step4_theme"]["learning_file"] = str(learning_path)
    except Exception as e:
        ctx["debug"]["step4_theme"]["learning_write"] = {"ok": False, "reason": f"exception:{type(e).__name__}"}

    return ctx


def run_step4_theme_boost(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧入口名。"""
    return run_step4(s, ctx)
