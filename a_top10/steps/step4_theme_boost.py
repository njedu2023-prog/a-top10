# -*- coding: utf-8 -*-
"""
Step4 : 题材/板块加权（ThemeBoost）

目标（0204-TOP10）：
- 输出字段：题材加成（0~1） / ThemeBoost（0~1）
- 先用 hot_boards.csv（行业热度）跑通最小闭环
- 若缺行业字段，自动从 stock_basic 补齐
- 全程写 debug：到底读到了什么、匹配了多少、为何为 0
- 入口：run_step4(s, ctx) -> Dict[str, Any]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

from a_top10.config import Settings


# -------------------------
# Config
# -------------------------
INDUSTRY_COL_CANDIDATES = [
    "industry",
    "industry_name",
    "industryName",
    "申万行业",
    "申万一级行业",
    "一级行业",
    "行业",
    "行业名称",
]

CODE_COL_CANDIDATES = ["ts_code", "code", "TS_CODE", "股票代码", "证券代码"]

HOT_BOARDS_INDUSTRY_COLS = ["industry", "industry_name", "行业", "板块", "板块名称", "行业名称"]
HOT_BOARDS_RANK_COLS = ["rank", "Rank", "排名", "hot_rank", "热度排名"]

# 龙虎榜：若 ctx 有 top_list（或 snap/top_list.csv），命中则加分
DRAGON_CODE_COLS = ["ts_code", "code", "TS_CODE", "股票代码", "证券代码"]

# 默认龙虎榜加成（你现在看到的 0.08 就是它）
DEFAULT_DRAGON_BONUS = 0.08

# 行业热度衰减参数：rank 越小分越高
# score = exp(-k*(rank-1))
DEFAULT_RANK_DECAY_K = 0.18

# 只取热榜前 N 个行业参与（避免长尾噪声）
DEFAULT_TOPK_INDUSTRY = 40


# -------------------------
# Helpers
# -------------------------
def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _parse_rank_int(x: Any, default: int = 9999) -> int:
    """
    强韧 rank 解析：
    支持： 1 / 1.0 / "01" / "第1名" / "1/10" / "No.1" / "1名"
    取第一个正整数。
    """
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return default
        return int(x)

    s = str(x).strip()
    if not s:
        return default

    # 常见形式：1/10 -> 1
    m = re.search(r"(\d+)", s)
    if not m:
        return default
    try:
        v = int(m.group(1))
        if v <= 0:
            return default
        return v
    except Exception:
        return default


def _norm_code(code: Any) -> Tuple[str, str]:
    """
    标准化股票代码：
    - 返回 (ts_code_upper, code6)
    支持：
    - 002506.SZ -> ("002506.SZ", "002506")
    - 002506 -> ("002506", "002506")
    """
    s = _safe_str(code).upper()
    if not s:
        return ("", "")
    code6 = s.split(".")[0]
    if len(code6) > 6 and code6.isdigit():
        code6 = code6[-6:]
    return (s, code6)


def _read_csv_guess(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception:
            pass
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


def _ensure_outputs_dir(ctx: Dict[str, Any], s: Optional[Settings]) -> Path:
    # 优先 ctx 里 outputs_dir
    p = _ctx_get_path(ctx, ["outputs_dir", "output_dir", "out_dir"])
    if p is not None:
        p.mkdir(parents=True, exist_ok=True)
        return p
    # 再尝试 settings
    if s is not None:
        try:
            od = getattr(s, "output", None)
            if od is not None and hasattr(od, "dir"):
                p2 = Path(od.dir)
                p2.mkdir(parents=True, exist_ok=True)
                return p2
        except Exception:
            pass
    # 兜底
    p3 = Path("outputs")
    p3.mkdir(parents=True, exist_ok=True)
    return p3


def _resolve_trade_date(ctx: Dict[str, Any]) -> str:
    # 多个可能字段
    for k in ("trade_date", "TRADE_DATE", "date", "run_date"):
        v = ctx.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 兜底：unknown
    return "unknown"


# -------------------------
# Core: compute ThemeBoost
# -------------------------
@dataclass
class ThemeDebug:
    hot_boards_rows: int = 0
    hot_boards_cols: List[str] = None
    hot_rank_col: str = ""
    hot_industry_col: str = ""
    industry_score_map_size: int = 0
    industry_score_min: float = 0.0
    industry_score_max: float = 0.0
    matched_industry_count: int = 0
    dragon_hits: int = 0
    theme_boost_nonzero: int = 0
    reason: str = ""


def _build_industry_score_map(
    hot_boards: pd.DataFrame,
    k: float = DEFAULT_RANK_DECAY_K,
    topk: int = DEFAULT_TOPK_INDUSTRY,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    从 hot_boards 构建：industry -> score(0..1)
    """
    dbg: Dict[str, Any] = {}
    if hot_boards is None or hot_boards.empty:
        dbg["ok"] = False
        dbg["reason"] = "hot_boards empty"
        return {}, dbg

    ind_col = _first_existing_col(hot_boards, HOT_BOARDS_INDUSTRY_COLS)
    rank_col = _first_existing_col(hot_boards, HOT_BOARDS_RANK_COLS)
    dbg["industry_col"] = ind_col or ""
    dbg["rank_col"] = rank_col or ""

    if not ind_col or not rank_col:
        dbg["ok"] = False
        dbg["reason"] = "hot_boards missing industry/rank col"
        return {}, dbg

    df = hot_boards[[ind_col, rank_col]].copy()
    df[ind_col] = df[ind_col].astype(str).map(lambda x: _safe_str(x))
    df[rank_col] = df[rank_col].map(lambda x: _parse_rank_int(x, default=9999))

    # 过滤无效
    df = df[(df[ind_col] != "") & (df[rank_col] < 9999)]
    if df.empty:
        dbg["ok"] = False
        dbg["reason"] = "hot_boards rank parse all invalid"
        return {}, dbg

    # 去重：同一行业取最好（rank最小）
    df = df.sort_values(rank_col, ascending=True)
    df = df.drop_duplicates(subset=[ind_col], keep="first")

    # 取 topk
    if topk and topk > 0:
        df = df.head(int(topk))

    # score
    ranks = df[rank_col].astype(int).values
    scores = np.exp(-k * (ranks - 1.0))
    # 归一化到 0..1（防止全部接近 1 或接近 0）
    # 如果 max==min，就保持原值但 clip
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if abs(s_max - s_min) > 1e-9:
        scores = (scores - s_min) / (s_max - s_min)
    scores = np.clip(scores, 0.0, 1.0)

    mp: Dict[str, float] = {}
    for ind, sc in zip(df[ind_col].tolist(), scores.tolist()):
        if ind:
            mp[str(ind).strip()] = float(sc)

    dbg["ok"] = True
    dbg["n"] = int(len(mp))
    dbg["score_min"] = float(np.min(list(mp.values()))) if mp else 0.0
    dbg["score_max"] = float(np.max(list(mp.values()))) if mp else 0.0
    dbg["top5"] = sorted(mp.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return mp, dbg


def _apply_industry_and_dragon(
    cand_df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    top_list: pd.DataFrame,
    industry_score: Dict[str, float],
    dragon_bonus: float = DEFAULT_DRAGON_BONUS,
) -> Tuple[pd.DataFrame, ThemeDebug]:
    """
    对候选df计算 ThemeBoost / 题材加成：
    - industry_boost：按行业热度映射
    - dragon_bonus：命中龙虎榜则加固定值（默认0.08）
    """
    dbg = ThemeDebug(hot_boards_cols=[],)

    if cand_df is None or cand_df.empty:
        dbg.reason = "candidate df empty"
        out = pd.DataFrame() if cand_df is None else cand_df.copy()
        return out, dbg

    out = cand_df.copy()

    # code 列
    code_col = _first_existing_col(out, CODE_COL_CANDIDATES)
    if not code_col:
        dbg.reason = "candidate missing code col"
        # 兜底：直接给 0
        out["ThemeBoost"] = 0.0
        out["题材加成"] = 0.0
        return out, dbg

    # industry 列：若没有，用 stock_basic 补
    ind_col = _first_existing_col(out, INDUSTRY_COL_CANDIDATES)
    if not ind_col and stock_basic is not None and not stock_basic.empty:
        sb_code_col = _first_existing_col(stock_basic, CODE_COL_CANDIDATES)
        sb_ind_col = _first_existing_col(stock_basic, INDUSTRY_COL_CANDIDATES)
        if sb_code_col and sb_ind_col:
            tmp = stock_basic[[sb_code_col, sb_ind_col]].copy()
            tmp["_code6"] = tmp[sb_code_col].map(lambda x: _norm_code(x)[1])
            tmp = tmp.dropna(subset=["_code6"]).drop_duplicates(subset=["_code6"], keep="first")

            out["_code6"] = out[code_col].map(lambda x: _norm_code(x)[1])
            out = out.merge(tmp[["_code6", sb_ind_col]], on="_code6", how="left")
            ind_col = sb_ind_col  # merge 后列名保持 sb_ind_col
        # 否则只能没有行业

    # 龙虎榜命中集合
    dragon_set6 = set()
    dragon_set_ts = set()
    if top_list is not None and not top_list.empty:
        tl_code_col = _first_existing_col(top_list, DRAGON_CODE_COLS)
        if tl_code_col:
            for v in top_list[tl_code_col].tolist():
                ts, c6 = _norm_code(v)
                if ts:
                    dragon_set_ts.add(ts)
                if c6:
                    dragon_set6.add(c6)

    # 计算 industry_boost
    industry_boost = np.zeros(len(out), dtype=float)
    matched = 0
    if ind_col and industry_score:
        inds = out[ind_col].astype(str).map(lambda x: _safe_str(x)).tolist()
        for i, ind in enumerate(inds):
            if not ind:
                continue
            sc = industry_score.get(ind, None)
            if sc is None:
                # 轻微容错：去空格/全角空格
                ind2 = re.sub(r"\s+", "", ind)
                sc = industry_score.get(ind2, None)
            if sc is not None:
                industry_boost[i] = float(sc)
                matched += 1

    dbg.matched_industry_count = int(matched)

    # 龙虎榜加成
    dragon_bonus_arr = np.zeros(len(out), dtype=float)
    dh = 0
    codes = out[code_col].tolist()
    for i, v in enumerate(codes):
        ts, c6 = _norm_code(v)
        hit = (ts in dragon_set_ts) or (c6 in dragon_set6)
        if hit:
            dragon_bonus_arr[i] = float(dragon_bonus)
            dh += 1
    dbg.dragon_hits = int(dh)

    theme = industry_boost + dragon_bonus_arr
    theme = np.clip(theme, 0.0, 1.0)

    out["ThemeBoost"] = theme.astype("float64")
    out["题材加成"] = out["ThemeBoost"]

    # 方便报告展示：输出板块/行业字段
    if ind_col and ind_col in out.columns:
        out["板块"] = out[ind_col].astype(str).map(lambda x: _safe_str(x))
    else:
        out["板块"] = ""

    dbg.theme_boost_nonzero = int((out["ThemeBoost"] > 0).sum())
    return out, dbg


# -------------------------
# Entry
# -------------------------
def run_step4(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入：
      ctx 中应有候选 df（通常 step3 输出），以及 hot_boards / stock_basic / top_list（可选）
    输出：
      ctx["theme_df"]：带 ThemeBoost/题材加成 的 df
      ctx["debug"]["step4_theme"]：可复盘诊断
      并尝试落盘 outputs/debug_step4_theme_YYYYMMDD.json
    """
    ctx = ctx or {}
    ctx.setdefault("debug", {})
    dbg_all: Dict[str, Any] = {}

    # 1) 取候选df（尽量兼容）
    cand_df = _ctx_get_df(ctx, ["step3_df", "candidates", "candidate_df", "df"])
    if cand_df is None or cand_df.empty:
        # 兜底：若 ctx 里有 step3 的结构
        v = ctx.get("step3", None)
        if isinstance(v, dict):
            cand_df = v.get("df", pd.DataFrame())
    if cand_df is None:
        cand_df = pd.DataFrame()

    # 2) hot_boards / stock_basic / top_list
    hot_boards = _ctx_get_df(ctx, ["hot_boards", "hot_boards_df"])
    stock_basic = _ctx_get_df(ctx, ["stock_basic", "stock_basic_df"])
    top_list = _ctx_get_df(ctx, ["top_list", "top_list_df", "龙虎榜", "longhu"])

    # 若 ctx 里没带 df，尝试从 snapshot_dir 读取
    snapshot_dir = _ctx_get_path(ctx, ["snapshot_dir", "snap_dir", "snapshot_path"])
    if snapshot_dir is not None:
        if hot_boards.empty:
            hot_boards = _read_csv_guess(Path(snapshot_dir) / "hot_boards.csv")
        if stock_basic.empty:
            stock_basic = _read_csv_guess(Path(snapshot_dir) / "stock_basic.csv")
        if top_list.empty:
            # 你仓库里有 top_list.csv（历史快照字段列表也包含 top_list.csv）
            top_list = _read_csv_guess(Path(snapshot_dir) / "top_list.csv")

    # 3) build industry score map
    industry_map, dbg_map = _build_industry_score_map(
        hot_boards,
        k=float(getattr(getattr(s, "weights", {}), "rank_decay_k", DEFAULT_RANK_DECAY_K))
        if s is not None else DEFAULT_RANK_DECAY_K,
        topk=DEFAULT_TOPK_INDUSTRY,
    )

    # 4) apply
    out_df, dbg_apply = _apply_industry_and_dragon(
        cand_df=cand_df,
        stock_basic=stock_basic,
        top_list=top_list,
        industry_score=industry_map,
        dragon_bonus=DEFAULT_DRAGON_BONUS,
    )

    # 5) assemble debug
    td = _resolve_trade_date(ctx)
    dbg_all["trade_date"] = td
    dbg_all["hot_boards_rows"] = int(len(hot_boards)) if isinstance(hot_boards, pd.DataFrame) else 0
    dbg_all["hot_boards_cols"] = list(hot_boards.columns) if isinstance(hot_boards, pd.DataFrame) else []
    dbg_all["industry_score_map"] = {
        "ok": bool(dbg_map.get("ok")),
        "reason": dbg_map.get("reason", ""),
        "industry_col": dbg_map.get("industry_col", ""),
        "rank_col": dbg_map.get("rank_col", ""),
        "size": int(dbg_map.get("n", 0)) if dbg_map.get("ok") else 0,
        "score_min": float(dbg_map.get("score_min", 0.0)),
        "score_max": float(dbg_map.get("score_max", 0.0)),
        "top5": dbg_map.get("top5", []),
    }

    dbg_all["matched_industry_count"] = int(getattr(dbg_apply, "matched_industry_count", 0))
    dbg_all["dragon_hits"] = int(getattr(dbg_apply, "dragon_hits", 0))
    dbg_all["theme_boost_nonzero"] = int(getattr(dbg_apply, "theme_boost_nonzero", 0))

    # 关键判断：为什么会“全是 0.08”
    if dbg_all["matched_industry_count"] == 0 and dbg_all["dragon_hits"] > 0:
        dbg_all["diagnosis"] = "industry heat NOT applied; only dragon bonus applied -> many 0.08"
    elif dbg_all["matched_industry_count"] == 0 and dbg_all["dragon_hits"] == 0:
        dbg_all["diagnosis"] = "both industry heat and dragon bonus NOT applied -> all 0"
    else:
        dbg_all["diagnosis"] = "industry heat applied (ok)"

    ctx["theme_df"] = out_df
    ctx["debug"]["step4_theme"] = dbg_all

    # 6) best-effort write debug file
    try:
        out_dir = _ensure_outputs_dir(ctx, s)
        p = out_dir / f"debug_step4_theme_{td}.json"
        p.write_text(json.dumps(dbg_all, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx["debug"]["step4_theme"]["debug_file"] = str(p)
    except Exception:
        # 不影响主链路
        pass

    return ctx


def run_step4_theme_boost(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧入口名（如果你的 main.py 曾经调用过这个）。"""
    return run_step4(s, ctx)
