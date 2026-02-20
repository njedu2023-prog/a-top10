# -*- coding: utf-8 -*-
"""
Step4 : 题材/板块加权（ThemeBoost）

目标（0204-TOP10）：
- 输出字段：题材加成（0~1） / ThemeBoost（0~1）
- 先用 hot_boards.csv（行业热度）跑通最小闭环
- 若缺行业字段，自动从 stock_basic 补齐
- 全程写 debug：到底读到了什么、匹配了多少、为何为 0
- ✅ 每天落库（GitHub Actions 工作区）：
    outputs/learning/step4_theme.csv
  目的：让 Step5 训练 → 自动通；Step7 学习 → 自动通
- ✅ 关键：保障 Step3 的核心特征列透传到 ctx["theme_df"]，供 Step5/feature_history 使用
- 入口：run_step4(s, ctx) -> Dict[str, Any]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    "证券简称代码",
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

HOT_BOARDS_INDUSTRY_COLS = ["industry", "industry_name", "行业", "板块", "板块名称", "行业名称"]
HOT_BOARDS_RANK_COLS = ["rank", "Rank", "排名", "hot_rank", "热度排名"]

DRAGON_CODE_COLS = [
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

DEFAULT_DRAGON_BONUS = 0.08
DEFAULT_RANK_DECAY_K = 0.18
DEFAULT_TOPK_INDUSTRY = 40
DEFAULT_FUZZY_MAX_ROWS = 300

STEP4_LEARNING_REL_PATH = Path("outputs") / "learning" / "step4_theme.csv"


# -------------------------
# Helpers
# -------------------------
_RE_FIRST_INT = re.compile(r"(\d+)")
_RE_CODE6 = re.compile(r"(\d{6})")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "<na>"):
        return ""
    return s


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
    s = _safe_str(code).upper()
    if not s:
        return ("", "")
    m = _RE_CODE6.search(s)
    code6 = m.group(1) if m else s.split(".")[0]
    return (s, code6)


def _norm_industry_key(x: Any) -> str:
    s = _safe_str(x)
    if not s:
        return ""
    s2 = re.sub(r"[\s\u3000]+", "", s)
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


def _read_csv_guess_any(path: Path) -> pd.DataFrame:
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
    base = _ensure_outputs_dir(ctx, s)
    p = base / "learning"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_trade_date(ctx: Dict[str, Any]) -> str:
    for k in ("trade_date", "TRADE_DATE", "date", "run_date"):
        v = ctx.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


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


def _get_rank_decay_k(s: Optional[Settings], default: float = DEFAULT_RANK_DECAY_K) -> float:
    try:
        return float(_get_weights_value(s, "rank_decay_k", default))
    except Exception:
        return float(default)


def _get_topk_industry(s: Optional[Settings], default: int = DEFAULT_TOPK_INDUSTRY) -> int:
    try:
        return int(_get_weights_value(s, "topk_industry", default))
    except Exception:
        return int(default)


def _get_dragon_bonus(s: Optional[Settings], default: float = DEFAULT_DRAGON_BONUS) -> float:
    try:
        return float(_get_weights_value(s, "dragon_bonus", default))
    except Exception:
        return float(default)


def _get_fuzzy_max_rows(s: Optional[Settings], default: int = DEFAULT_FUZZY_MAX_ROWS) -> int:
    try:
        return int(_get_weights_value(s, "fuzzy_max_rows", default))
    except Exception:
        return int(default)


def _series_nonblank_ratio(sr: Optional[pd.Series]) -> float:
    if sr is None or len(sr) == 0:
        return 0.0
    vals = sr.astype(str).map(_safe_str)
    return float((vals != "").mean())


def _series_nonzero_ratio(sr: Optional[pd.Series]) -> float:
    if sr is None or len(sr) == 0:
        return 0.0
    x = pd.to_numeric(sr, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((x != 0.0).mean())


# -------------------------
# ✅ Step5 特征透传保障（关键修复）
# -------------------------
_STEP5_FEATURE_ALIASES: Dict[str, Sequence[str]] = {
    "StrengthScore": ["StrengthScore", "strengthscore", "strength_score", "Strength", "strength", "强度得分", "强度", "强度分"],
    "ThemeBoost": ["ThemeBoost", "themeboost", "theme_boost", "题材加成", "题材", "Theme"],
    "seal_amount": ["seal_amount", "sealamount", "seal_amt", "seal", "封单额", "封单金额", "封单"],
    "open_times": ["open_times", "opentimes", "open_time", "open_count", "openings", "打开次数", "开板次数", "炸板次数"],
    "turnover_rate": ["turnover_rate", "turnoverrate", "turn_rate", "turnover", "换手率", "换手率%"],
}


def _ensure_step5_features_passthrough(df: pd.DataFrame, dbg: Dict[str, Any]) -> pd.DataFrame:
    """
    保障 Step4 输出 df 中至少存在 Step5 训练/推断的核心特征列，并尽量把别名映射到标准名。
    ✅ 同时把这些列转成数值（空串->0），避免 Step5 写 feature_history 时被动变成 0/NaN。
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

    # 数值化（关键）：避免下游拿到 object 空字符串
    for c in ("StrengthScore", "ThemeBoost", "seal_amount", "open_times", "turnover_rate"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

    # 兼容 writers/旧逻辑
    if "题材加成" not in out.columns and "ThemeBoost" in out.columns:
        out["题材加成"] = out["ThemeBoost"]
    if "board" not in out.columns and "板块" in out.columns:
        out["board"] = out["板块"]

    dbg["step5_feature_passthrough"] = {"mapped": mapped, "missing": missing}

    # ✅ 加一段可验证的非零率（锁定“是不是 Step4 丢字段”）
    dbg["theme_df_feature_nonzero_rate"] = {
        "StrengthScore": _series_nonzero_ratio(out.get("StrengthScore")),
        "turnover_rate": _series_nonzero_ratio(out.get("turnover_rate")),
        "seal_amount": _series_nonzero_ratio(out.get("seal_amount")),
        "open_times": _series_nonzero_ratio(out.get("open_times")),
    }
    return out


# -------------------------
# Core: compute ThemeBoost
# -------------------------
@dataclass
class ThemeDebug:
    hot_boards_rows: int = 0
    hot_boards_cols: List[str] = field(default_factory=list)
    hot_rank_col: str = ""
    hot_industry_col: str = ""
    industry_score_map_size: int = 0
    industry_score_min: float = 0.0
    industry_score_max: float = 0.0

    candidate_rows: int = 0
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

    industry_map_empty: bool = False
    top_list_rows: int = 0
    stock_basic_rows: int = 0


def _build_industry_score_map(
    hot_boards: pd.DataFrame,
    k: float = DEFAULT_RANK_DECAY_K,
    topk: int = DEFAULT_TOPK_INDUSTRY,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
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
    if topk and int(topk) > 0:
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


def _enrich_industry_from_stock_basic(
    out: pd.DataFrame,
    code_col: str,
    ind_col: Optional[str],
    stock_basic: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, float, float]:
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
    fuzzy_max_rows: int = DEFAULT_FUZZY_MAX_ROWS,
) -> Tuple[pd.DataFrame, ThemeDebug]:
    dbg = ThemeDebug()

    if cand_df is None or cand_df.empty:
        dbg.reason = "candidate df empty"
        out = pd.DataFrame() if cand_df is None else cand_df.copy()
        return out, dbg

    out = cand_df.copy()
    dbg.candidate_rows = int(len(out))
    dbg.stock_basic_rows = int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0
    dbg.top_list_rows = int(len(top_list)) if isinstance(top_list, pd.DataFrame) else 0

    code_col = _first_existing_col(out, CODE_COL_CANDIDATES)
    dbg.candidate_code_col = code_col or ""
    if not code_col:
        dbg.reason = "candidate missing code col"
        out["ThemeBoost"] = 0.0
        out["题材加成"] = 0.0
        out["板块"] = ""
        return out, dbg

    if "ts_code" not in out.columns:
        out["ts_code"] = out[code_col].astype(str).map(_safe_str)

    ind_col_before = _first_existing_col(out, INDUSTRY_COL_CANDIDATES)
    dbg.candidate_industry_col_before = ind_col_before or ""

    out, ind_col_final, ratio_before, ratio_after = _enrich_industry_from_stock_basic(
        out=out, code_col=code_col, ind_col=ind_col_before, stock_basic=stock_basic
    )
    dbg.candidate_industry_col_final = ind_col_final or ""
    dbg.candidate_industry_nonblank_ratio_before = float(ratio_before)
    dbg.candidate_industry_nonblank_ratio_final = float(ratio_after)
    ind_col = ind_col_final if ind_col_final else None

    dragon_ts_set: set = set()
    dragon_c6_set: set = set()
    if top_list is not None and not top_list.empty:
        tl_code_col = _first_existing_col(top_list, DRAGON_CODE_COLS)
        if tl_code_col:
            tl = top_list[tl_code_col].astype(str).map(_safe_str)
            dragon_ts = tl.map(lambda x: _norm_code(x)[0])
            dragon_c6 = tl.map(lambda x: _norm_code(x)[1])
            dragon_ts = dragon_ts[dragon_ts != ""].drop_duplicates()
            dragon_c6 = dragon_c6[dragon_c6 != ""].drop_duplicates()
            dragon_ts_set = set(dragon_ts.tolist())
            dragon_c6_set = set(dragon_c6.tolist())

    industry_map_exact = dict(industry_score or {})
    dbg.industry_map_empty = (len(industry_map_exact) == 0)

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

        exact_sc = inds_raw.map(industry_map_exact)
        m_exact = exact_sc.notna()
        if m_exact.any():
            industry_boost[m_exact.to_numpy()] = exact_sc[m_exact].astype(float).to_numpy()
            c = int(m_exact.sum())
            matched += c
            matched_exact += c

        remain = ~m_exact
        if remain.any():
            norm_sc = inds_norm[remain].map(industry_map_norm)
            m_norm = norm_sc.notna()
            if m_norm.any():
                remain_idx = np.where(remain.to_numpy())[0]
                hit_pos = remain_idx[m_norm.to_numpy()]
                industry_boost[hit_pos] = norm_sc[m_norm].astype(float).to_numpy()
                c = int(m_norm.sum())
                matched += c
                matched_norm += c

        remain2_mask = (industry_boost == 0.0) & (inds_norm != "")
        remain2_pos = np.where(remain2_mask)[0]
        if len(remain2_pos) > 0 and norm_keys:
            if fuzzy_max_rows and len(remain2_pos) > int(fuzzy_max_rows):
                remain2_pos = remain2_pos[: int(fuzzy_max_rows)]

            for pos in remain2_pos:
                nk = inds_norm.iat[pos]
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
                    industry_boost[pos] = float(best)
                    matched += 1
                    matched_fuzzy += 1

    dbg.matched_industry_count = int(matched)
    dbg.matched_industry_exact = int(matched_exact)
    dbg.matched_industry_norm = int(matched_norm)
    dbg.matched_industry_fuzzy = int(matched_fuzzy)

    codes = out[code_col].astype(str).map(_safe_str)
    codes_ts = codes.map(lambda x: _norm_code(x)[0])
    codes_c6 = codes.map(lambda x: _norm_code(x)[1])

    if dragon_ts_set or dragon_c6_set:
        hit_ts = codes_ts.isin(dragon_ts_set)
        hit_c6 = codes_c6.isin(dragon_c6_set)
        hit = (hit_ts | hit_c6).to_numpy()
    else:
        hit = np.zeros(n, dtype=bool)

    dragon_bonus_arr = np.zeros(n, dtype=float)
    if hit.any():
        dragon_bonus_arr[hit] = float(dragon_bonus)
    dbg.dragon_hits = int(hit.sum())

    theme = np.clip(industry_boost + dragon_bonus_arr, 0.0, 1.0)

    out["ThemeBoost"] = theme.astype("float64")
    out["题材加成"] = out["ThemeBoost"]

    if ind_col and ind_col in out.columns:
        out["板块"] = out[ind_col].astype(str).map(_safe_str)
    else:
        out["板块"] = ""

    dbg.theme_boost_nonzero = int((out["ThemeBoost"] > 0).sum())

    if dbg.industry_map_empty:
        dbg.reason = "industry map empty (hot_boards missing/empty or parse failed)"
    elif not ind_col or ind_col not in out.columns:
        dbg.reason = "candidate industry column missing (and enrich failed or not available)"
    elif dbg.candidate_industry_nonblank_ratio_final < 0.1:
        dbg.reason = "candidate industry mostly blank (after enrich), industry match likely 0"
    elif dbg.matched_industry_count == 0:
        dbg.reason = "industry match 0 (names mismatch), consider normalize/fuzzy or check hot_boards industry values"

    return out, dbg


# -------------------------
# Learning DB (GitHub outputs/learning/step4_theme.csv)
# -------------------------
def _build_step4_learning_frame(out_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    ✅ 注意：step4_theme.csv 是“题材落库表”，不是全量特征表
    只固定落：run_time_utc/trade_date/ts_code/name/ThemeBoost/板块
    """
    if out_df is None or out_df.empty:
        return pd.DataFrame(columns=["run_time_utc", "trade_date", "ts_code", "name", "ThemeBoost", "板块"])

    df = out_df.copy()

    code_col = _first_existing_col(df, CODE_COL_CANDIDATES) or ""
    name_col = _first_existing_col(df, NAME_COL_CANDIDATES) or ""

    if not code_col:
        return pd.DataFrame(columns=["run_time_utc", "trade_date", "ts_code", "name", "ThemeBoost", "板块"])

    run_time_utc = _utc_now_iso()

    codes = df[code_col].astype(str).map(_safe_str)
    ts_code = codes.map(lambda x: _norm_code(x)[0])
    ts_code = ts_code.where(ts_code != "", codes.map(lambda x: _norm_code(x)[1]))

    name = df[name_col].astype(str).map(_safe_str) if name_col else ""
    theme = df["ThemeBoost"] if "ThemeBoost" in df.columns else 0.0
    board = df["板块"].astype(str).map(_safe_str) if "板块" in df.columns else ""

    out = pd.DataFrame(
        {
            "run_time_utc": run_time_utc,
            "trade_date": _safe_str(trade_date),
            "ts_code": ts_code.astype(str).map(_safe_str),
            "name": name if isinstance(name, pd.Series) else "",
            "ThemeBoost": pd.to_numeric(theme, errors="coerce").fillna(0.0).astype("float64"),
            "板块": board,
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

    old = _read_csv_guess_any(learning_path)
    dbg["rows_old"] = int(len(old)) if isinstance(old, pd.DataFrame) else 0

    merged = learn_df.copy() if (old is None or old.empty) else pd.concat([old, learn_df], ignore_index=True, sort=False)

    for c in ["run_time_utc", "trade_date", "ts_code", "name", "ThemeBoost", "板块"]:
        if c not in merged.columns:
            merged[c] = ""

    merged["trade_date"] = merged["trade_date"].astype(str).map(_safe_str)
    merged["ts_code"] = merged["ts_code"].astype(str).map(_safe_str)
    merged["run_time_utc"] = merged["run_time_utc"].astype(str).map(_safe_str)

    merged = merged.sort_values(["trade_date", "ts_code", "run_time_utc"], ascending=[True, True, True])
    merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

    merged["ThemeBoost"] = pd.to_numeric(merged["ThemeBoost"], errors="coerce").fillna(0.0).astype("float64")
    merged = merged.sort_values(["trade_date", "ThemeBoost"], ascending=[True, False]).reset_index(drop=True)

    _atomic_write_csv(merged, learning_path)

    dbg["ok"] = True
    dbg["rows_new_total"] = int(len(merged))
    return dbg


# -------------------------
# Entry
# -------------------------
def run_step4(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入：
      ctx 中应有候选 df（通常 step3 输出）
    输出：
      ctx["theme_df"]：带 ThemeBoost/题材加成 的 df（且 ✅ 透传 Step5 需要特征列）
      ctx["debug"]["step4_theme"]：可复盘诊断
      并尝试落盘：
        outputs/debug_step4_theme_YYYYMMDD.json
      ✅ 并尝试落库：
        outputs/learning/step4_theme.csv
    """
    ctx = ctx or {}
    ctx.setdefault("debug", {})
    dbg_all: Dict[str, Any] = {}

    # 1) 取候选df（加强兼容：优先拿“step3_df / step3_strength / ctx['step3']”）
    cand_df = _ctx_get_df(ctx, ["step3_df", "step3_strength_df", "step3_out", "candidates", "candidate_df", "df"])
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

    snapshot_dir = _ctx_get_path(ctx, ["snapshot_dir", "snap_dir", "snapshot_path"])
    snapshot_files: Dict[str, str] = {}
    if snapshot_dir is not None:
        sd = Path(snapshot_dir)
        if hot_boards.empty:
            p = sd / "hot_boards.csv"
            snapshot_files["hot_boards.csv"] = str(p)
            hot_boards = _read_csv_guess(p)
        if stock_basic.empty:
            p = sd / "stock_basic.csv"
            snapshot_files["stock_basic.csv"] = str(p)
            stock_basic = _read_csv_guess(p)
        if top_list.empty:
            p = sd / "top_list.csv"
            snapshot_files["top_list.csv"] = str(p)
            top_list = _read_csv_guess(p)

    # 3) build industry score map
    topk_industry = _get_topk_industry(s, DEFAULT_TOPK_INDUSTRY)
    rank_decay_k = _get_rank_decay_k(s, DEFAULT_RANK_DECAY_K)
    industry_map, dbg_map = _build_industry_score_map(hot_boards=hot_boards, k=rank_decay_k, topk=topk_industry)

    # 4) apply
    dragon_bonus = _get_dragon_bonus(s, DEFAULT_DRAGON_BONUS)
    fuzzy_max_rows = _get_fuzzy_max_rows(s, DEFAULT_FUZZY_MAX_ROWS)

    out_df, dbg_apply = _apply_industry_and_dragon(
        cand_df=cand_df,
        stock_basic=stock_basic,
        top_list=top_list,
        industry_score=industry_map,
        dragon_bonus=dragon_bonus,
        fuzzy_max_rows=fuzzy_max_rows,
    )

    # 5) assemble debug
    td = _resolve_trade_date(ctx)
    dbg_all["trade_date"] = td
    dbg_all["snapshot_dir"] = str(snapshot_dir) if snapshot_dir is not None else ""
    dbg_all["snapshot_files"] = snapshot_files

    dbg_all["params"] = {
        "rank_decay_k": float(rank_decay_k),
        "topk_industry": int(topk_industry),
        "dragon_bonus": float(dragon_bonus),
        "fuzzy_max_rows": int(fuzzy_max_rows),
    }

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
        "rows": int(getattr(dbg_apply, "candidate_rows", 0)),
        "code_col": getattr(dbg_apply, "candidate_code_col", ""),
        "industry_col_before": getattr(dbg_apply, "candidate_industry_col_before", ""),
        "industry_col_final": getattr(dbg_apply, "candidate_industry_col_final", ""),
        "industry_nonblank_ratio_before": float(getattr(dbg_apply, "candidate_industry_nonblank_ratio_before", 0.0)),
        "industry_nonblank_ratio_final": float(getattr(dbg_apply, "candidate_industry_nonblank_ratio_final", 0.0)),
        "stock_basic_rows": int(getattr(dbg_apply, "stock_basic_rows", 0)),
        "top_list_rows": int(getattr(dbg_apply, "top_list_rows", 0)),
    }

    dbg_all["matched_industry_count"] = int(getattr(dbg_apply, "matched_industry_count", 0))
    dbg_all["dragon_hits"] = int(getattr(dbg_apply, "dragon_hits", 0))
    dbg_all["theme_boost_nonzero"] = int(getattr(dbg_apply, "theme_boost_nonzero", 0))
    dbg_all["apply_reason"] = getattr(dbg_apply, "reason", "")

    dbg_all["matched_industry_detail"] = {
        "exact": int(getattr(dbg_apply, "matched_industry_exact", 0)),
        "normalized": int(getattr(dbg_apply, "matched_industry_norm", 0)),
        "fuzzy_contains": int(getattr(dbg_apply, "matched_industry_fuzzy", 0)),
    }

    # diagnosis
    if not bool(dbg_map.get("ok")):
        dbg_all["diagnosis"] = f"industry score map build failed: {dbg_map.get('reason','')}"
    elif dbg_all["matched_industry_count"] == 0 and dbg_all["dragon_hits"] > 0:
        dbg_all["diagnosis"] = "industry heat NOT applied; only dragon bonus applied -> many 0.08"
    elif dbg_all["matched_industry_count"] == 0 and dbg_all["dragon_hits"] == 0:
        final_ratio = dbg_all["candidate"].get("industry_nonblank_ratio_final", 0.0)
        if final_ratio < 0.1:
            dbg_all["diagnosis"] = "both industry heat and dragon bonus NOT applied -> likely industry blank (after enrich) and no dragon hits"
        else:
            dbg_all["diagnosis"] = (
                "both industry heat and dragon bonus NOT applied -> likely industry name mismatch "
                "(check hot_boards industry vs candidate industry)"
            )
    else:
        dbg_all["diagnosis"] = "industry heat applied (ok)"

    # ✅ 关键修复：保障 Step5 需要特征列透传到 theme_df
    out_df = _ensure_step5_features_passthrough(out_df, dbg_all)

    ctx["theme_df"] = out_df
    ctx["debug"]["step4_theme"] = dbg_all

    # 6) write debug file
    try:
        out_dir = _ensure_outputs_dir(ctx, s)
        p = out_dir / f"debug_step4_theme_{td}.json"
        p.write_text(json.dumps(dbg_all, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx["debug"]["step4_theme"]["debug_file"] = str(p)
    except Exception:
        pass

    # 7) 落库 step4_theme.csv（题材表）
    try:
        learning_dir = _ensure_learning_dir(ctx, s)
        learning_path = learning_dir / "step4_theme.csv"
        learn_df = _build_step4_learning_frame(out_df=out_df, trade_date=td)
        dbg_write = _upsert_step4_learning(learn_df, learning_path)
        ctx["debug"]["step4_theme"]["learning_write"] = dbg_write
        ctx["debug"]["step4_theme"]["learning_file"] = str(learning_path)
    except Exception as e:
        ctx["debug"]["step4_theme"]["learning_write"] = {"ok": False, "reason": f"exception: {type(e).__name__}"}

    return ctx


def run_step4_theme_boost(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧入口名（如果你的 main.py 曾经调用过这个）。"""
    return run_step4(s, ctx)
