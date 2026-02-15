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
    "板块",
    "板块名称",
]

# 兼容更多常见“代码列”命名（尤其是 tushare stock_basic 的 symbol）
CODE_COL_CANDIDATES = [
    "ts_code",
    "code",
    "TS_CODE",
    "股票代码",
    "证券代码",
    "symbol",
    "Symbol",
    "ticker",
    "sec_code",
    "证券简称代码",
]

HOT_BOARDS_INDUSTRY_COLS = ["industry", "industry_name", "行业", "板块", "板块名称", "行业名称"]
HOT_BOARDS_RANK_COLS = ["rank", "Rank", "排名", "hot_rank", "热度排名"]

# 龙虎榜：若 ctx 有 top_list（或 snap/top_list.csv），命中则加分
DRAGON_CODE_COLS = [
    "ts_code",
    "code",
    "TS_CODE",
    "股票代码",
    "证券代码",
    "symbol",
    "Symbol",
    "ticker",
    "sec_code",
]

DEFAULT_DRAGON_BONUS = 0.08
DEFAULT_RANK_DECAY_K = 0.18
DEFAULT_TOPK_INDUSTRY = 40


# -------------------------
# Helpers
# -------------------------
_RE_FIRST_INT = re.compile(r"(\d+)")
_RE_CODE6 = re.compile(r"(\d{6})")


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


import pandas as pd
import numpy as np

def _safe_str(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "<na>"):
        return ""
    return s


def _parse_rank_int(x: Any, default: int = 9999) -> int:
    """
    强韧 rank 解析：
    支持： 1 / 1.0 / "01" / "第1名" / "1/10" / "No.1" / "1名"
    取第一个正整数。
    """
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

    s = str(x).strip()
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


def _norm_code(code: Any) -> Tuple[str, str]:
    """
    标准化股票代码：
    - 返回 (ts_code_upper, code6)
    支持：
    - 002506.SZ -> ("002506.SZ", "002506")
    - 002506 -> ("002506", "002506")
    - "SZ002506" / "sh600410" 之类的也尽量提取 6 位数字
    """
    s = _safe_str(code).upper()
    if not s:
        return ("", "")

    m = _RE_CODE6.search(s)
    code6 = m.group(1) if m else s.split(".")[0]
    return (s, code6)


def _norm_industry_key(x: Any) -> str:
    """
    行业/板块名规范化：
    - 去空白（含全角空格）
    - 去常见后缀词（行业/板块/概念）
    - 小写化
    """
    s = _safe_str(x)
    if not s:
        return ""
    s2 = re.sub(r"[\s\u3000]+", "", s)  # 空格/制表/全角空格
    for suf in ("行业", "板块", "概念"):
        if s2.endswith(suf) and len(s2) > len(suf):
            s2 = s2[: -len(suf)]
    return s2.lower()


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
    p = _ctx_get_path(ctx, ["outputs_dir", "output_dir", "out_dir"])
    if p is not None:
        p.mkdir(parents=True, exist_ok=True)
        return p

    if s is not None:
        try:
            od = getattr(s, "output", None)
            if od is not None and hasattr(od, "dir"):
                p2 = Path(od.dir)
                p2.mkdir(parents=True, exist_ok=True)
                return p2
        except Exception:
            pass

    p3 = Path("outputs")
    p3.mkdir(parents=True, exist_ok=True)
    return p3


def _resolve_trade_date(ctx: Dict[str, Any]) -> str:
    for k in ("trade_date", "TRADE_DATE", "date", "run_date"):
        v = ctx.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def _get_rank_decay_k(s: Optional[Settings], default: float = DEFAULT_RANK_DECAY_K) -> float:
    """
    兼容 Settings.weights 既可能是对象也可能是 dict 的情况。
    """
    if s is None:
        return float(default)
    w = getattr(s, "weights", None)
    if w is None:
        return float(default)
    try:
        if isinstance(w, dict):
            v = w.get("rank_decay_k", default)
            return float(v)
        v2 = getattr(w, "rank_decay_k", default)
        return float(v2)
    except Exception:
        return float(default)


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

    candidate_code_col: str = ""
    candidate_industry_col_before: str = ""
    candidate_industry_col_final: str = ""
    candidate_industry_nonblank_ratio_before: float = 0.0
    candidate_industry_nonblank_ratio_final: float = 0.0

    matched_industry_count: int = 0
    dragon_hits: int = 0
    theme_boost_nonzero: int = 0
    reason: str = ""

    matched_industry_exact: int = 0
    matched_industry_norm: int = 0
    matched_industry_fuzzy: int = 0


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
    df[ind_col] = df[ind_col].astype(str).map(_safe_str)
    df[rank_col] = df[rank_col].map(lambda x: _parse_rank_int(x, default=9999))

    df = df[(df[ind_col] != "") & (df[rank_col] < 9999)]
    if df.empty:
        dbg["ok"] = False
        dbg["reason"] = "hot_boards rank parse all invalid"
        return {}, dbg

    df = df.sort_values(rank_col, ascending=True).drop_duplicates(subset=[ind_col], keep="first")
    if topk and topk > 0:
        df = df.head(int(topk))

    ranks = df[rank_col].astype(int).to_numpy()
    scores = np.exp(-float(k) * (ranks - 1.0))

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if abs(s_max - s_min) > 1e-12:
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


def _series_nonblank_ratio(sr: Optional[pd.Series]) -> float:
    if sr is None or len(sr) == 0:
        return 0.0
    vals = sr.astype(str).map(_safe_str)
    return float((vals != "").mean())


def _enrich_industry_from_stock_basic(
    out: pd.DataFrame,
    code_col: str,
    ind_col: Optional[str],
    stock_basic: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, float, float]:
    """
    若 ind_col 缺失或“几乎全空”，从 stock_basic 按 code6 补齐一个最终行业列 _industry_final。
    只在必要时做，避免覆盖已有有效行业信息。
    """
    before_ratio = _series_nonblank_ratio(out[ind_col]) if ind_col and ind_col in out.columns else 0.0
    need_enrich = (not ind_col) or (before_ratio < 0.1)

    if not need_enrich:
        return out, ind_col or "", before_ratio, before_ratio

    if stock_basic is None or stock_basic.empty:
        return out, ind_col or "", before_ratio, before_ratio

    sb_code_col = _first_existing_col(stock_basic, CODE_COL_CANDIDATES)
    sb_ind_col = _first_existing_col(stock_basic, INDUSTRY_COL_CANDIDATES)
    if not sb_code_col or not sb_ind_col:
        return out, ind_col or "", before_ratio, before_ratio

    tmp = stock_basic[[sb_code_col, sb_ind_col]].copy()
    tmp["_code6"] = tmp[sb_code_col].map(lambda x: _norm_code(x)[1])
    tmp = tmp[tmp["_code6"].astype(str).map(_safe_str) != ""]
    tmp = tmp.drop_duplicates(subset=["_code6"], keep="first")

    out2 = out.copy()
    out2["_code6"] = out2[code_col].map(lambda x: _norm_code(x)[1])

    out2 = out2.merge(tmp[["_code6", sb_ind_col]], on="_code6", how="left")

    # 组装最终行业列：优先候选原行业，其次 stock_basic 行业
    out2["_industry_final"] = ""
    if ind_col and ind_col in out2.columns:
        out2["_industry_final"] = out2[ind_col].astype(str).map(_safe_str)

    out2["_industry_from_sb"] = out2[sb_ind_col].astype(str).map(_safe_str)
    out2["_industry_final"] = out2["_industry_final"].where(out2["_industry_final"] != "", out2["_industry_from_sb"])

    after_ratio = _series_nonblank_ratio(out2["_industry_final"])
    return out2, "_industry_final", before_ratio, after_ratio


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
    dbg = ThemeDebug(hot_boards_cols=[])

    if cand_df is None or cand_df.empty:
        dbg.reason = "candidate df empty"
        out = pd.DataFrame() if cand_df is None else cand_df.copy()
        return out, dbg

    out = cand_df.copy()

    # code 列
    code_col = _first_existing_col(out, CODE_COL_CANDIDATES)
    dbg.candidate_code_col = code_col or ""
    if not code_col:
        dbg.reason = "candidate missing code col"
        out["ThemeBoost"] = 0.0
        out["题材加成"] = 0.0
        out["板块"] = ""
        return out, dbg

    # industry 列（先看候选有没有）
    ind_col_before = _first_existing_col(out, INDUSTRY_COL_CANDIDATES)
    dbg.candidate_industry_col_before = ind_col_before or ""

    # 必要时用 stock_basic 补齐行业（不覆盖已有有效行业）
    out, ind_col_final, ratio_before, ratio_after = _enrich_industry_from_stock_basic(
        out=out, code_col=code_col, ind_col=ind_col_before, stock_basic=stock_basic
    )
    dbg.candidate_industry_col_final = ind_col_final or ""
    dbg.candidate_industry_nonblank_ratio_before = float(ratio_before)
    dbg.candidate_industry_nonblank_ratio_final = float(ratio_after)

    ind_col = ind_col_final if ind_col_final else None

    # 龙虎榜命中集合（向量化：ts_code / code6）
    dragon_ts = pd.Series(dtype=str)
    dragon_c6 = pd.Series(dtype=str)
    if top_list is not None and not top_list.empty:
        tl_code_col = _first_existing_col(top_list, DRAGON_CODE_COLS)
        if tl_code_col:
            tl = top_list[tl_code_col].astype(str).map(_safe_str)
            dragon_ts = tl.map(lambda x: _norm_code(x)[0])
            dragon_c6 = tl.map(lambda x: _norm_code(x)[1])
            dragon_ts = dragon_ts[dragon_ts != ""].drop_duplicates()
            dragon_c6 = dragon_c6[dragon_c6 != ""].drop_duplicates()

    # 行业热度映射：精确 + 规范化
    industry_map_exact = dict(industry_score or {})

    industry_map_norm: Dict[str, float] = {}
    for k, v in industry_map_exact.items():
        nk = _norm_industry_key(k)
        if nk:
            fv = float(v)
            if nk not in industry_map_norm or fv > float(industry_map_norm[nk]):
                industry_map_norm[nk] = fv

    norm_keys = list(industry_map_norm.keys())

    n = len(out)
    industry_boost = np.zeros(n, dtype=float)

    matched = 0
    matched_exact = 0
    matched_norm = 0
    matched_fuzzy = 0

    if ind_col and (industry_map_exact or industry_map_norm) and ind_col in out.columns:
        inds_raw = out[ind_col].astype(str).map(_safe_str)
        inds_norm = inds_raw.map(_norm_industry_key)

        # 1) 精确匹配（原文）
        exact_sc = inds_raw.map(industry_map_exact)
        m_exact = exact_sc.notna()
        if m_exact.any():
            industry_boost[m_exact.to_numpy()] = exact_sc[m_exact].astype(float).to_numpy()
            c = int(m_exact.sum())
            matched += c
            matched_exact += c

        # 2) 规范化匹配（只对未命中者）
        remain = ~m_exact
        if remain.any():
            norm_sc = inds_norm[remain].map(industry_map_norm)
            m_norm = norm_sc.notna()
            if m_norm.any():
                idx = norm_sc[m_norm].index
                industry_boost[out.index.get_indexer(idx)] = norm_sc[m_norm].astype(float).to_numpy()
                c = int(m_norm.sum())
                matched += c
                matched_norm += c

        # 3) fuzzy contains（只对仍未命中的少量行做逐行，topK<=40 性能可控）
        #    条件：nk 包含 kk 或 kk 包含 nk，取分最高的 kk
        remain2_mask = (industry_boost == 0.0) & (inds_norm != "")
        if remain2_mask.any() and norm_keys:
            remain2_idx = out.index[remain2_mask]
            for ridx in remain2_idx:
                nk = inds_norm.loc[ridx]
                if not nk:
                    continue
                best = None
                for kk in norm_keys:
                    if not kk:
                        continue
                    if (kk in nk) or (nk in kk):
                        vv = industry_map_norm.get(kk)
                        if vv is None:
                            continue
                        if best is None or float(vv) > float(best):
                            best = float(vv)
                if best is not None:
                    industry_boost[out.index.get_loc(ridx)] = float(best)
                    matched += 1
                    matched_fuzzy += 1

    dbg.matched_industry_count = int(matched)
    dbg.matched_industry_exact = int(matched_exact)
    dbg.matched_industry_norm = int(matched_norm)
    dbg.matched_industry_fuzzy = int(matched_fuzzy)

    # 龙虎榜加成（向量化）
    codes = out[code_col].astype(str).map(_safe_str)
    codes_ts = codes.map(lambda x: _norm_code(x)[0])
    codes_c6 = codes.map(lambda x: _norm_code(x)[1])

    hit_ts = codes_ts.isin(set(dragon_ts.tolist())) if len(dragon_ts) else pd.Series([False] * n, index=out.index)
    hit_c6 = codes_c6.isin(set(dragon_c6.tolist())) if len(dragon_c6) else pd.Series([False] * n, index=out.index)
    hit = (hit_ts | hit_c6).to_numpy()

    dragon_bonus_arr = np.zeros(n, dtype=float)
    if hit.any():
        dragon_bonus_arr[hit] = float(dragon_bonus)
    dbg.dragon_hits = int(hit.sum())

    theme = np.clip(industry_boost + dragon_bonus_arr, 0.0, 1.0)

    out["ThemeBoost"] = theme.astype("float64")
    out["题材加成"] = out["ThemeBoost"]

    # 报告展示：输出板块/行业字段（最终行业列）
    if ind_col and ind_col in out.columns:
        out["板块"] = out[ind_col].astype(str).map(_safe_str)
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
            top_list = _read_csv_guess(Path(snapshot_dir) / "top_list.csv")

    # 3) build industry score map
    industry_map, dbg_map = _build_industry_score_map(
        hot_boards=hot_boards,
        k=_get_rank_decay_k(s, DEFAULT_RANK_DECAY_K),
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

    dbg_all["candidate"] = {
        "code_col": getattr(dbg_apply, "candidate_code_col", ""),
        "industry_col_before": getattr(dbg_apply, "candidate_industry_col_before", ""),
        "industry_col_final": getattr(dbg_apply, "candidate_industry_col_final", ""),
        "industry_nonblank_ratio_before": float(getattr(dbg_apply, "candidate_industry_nonblank_ratio_before", 0.0)),
        "industry_nonblank_ratio_final": float(getattr(dbg_apply, "candidate_industry_nonblank_ratio_final", 0.0)),
    }

    dbg_all["matched_industry_count"] = int(getattr(dbg_apply, "matched_industry_count", 0))
    dbg_all["dragon_hits"] = int(getattr(dbg_apply, "dragon_hits", 0))
    dbg_all["theme_boost_nonzero"] = int(getattr(dbg_apply, "theme_boost_nonzero", 0))

    dbg_all["matched_industry_detail"] = {
        "exact": int(getattr(dbg_apply, "matched_industry_exact", 0)),
        "normalized": int(getattr(dbg_apply, "matched_industry_norm", 0)),
        "fuzzy_contains": int(getattr(dbg_apply, "matched_industry_fuzzy", 0)),
    }

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
        pass

    return ctx


def run_step4_theme_boost(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧入口名（如果你的 main.py 曾经调用过这个）。"""
    return run_step4(s, ctx)
