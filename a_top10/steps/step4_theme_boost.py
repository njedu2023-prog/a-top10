#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step4 : 题材/板块加权（ThemeBoost）

输入：
    strength_df（step3输出）
    + 可选：s（Settings/ctx），用于拿 step0 的 hot_boards / 龙虎榜等

输出：
    theme_df（附带 ThemeBoost + 分项）
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, List

import numpy as np
import pandas as pd


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


def _to_float_series(df: pd.DataFrame, col: Optional[str], default: float) -> pd.Series:
    if (col is None) or (col not in df.columns):
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s


def _ensure_ts_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts_col = _first_existing_col(out, ["ts_code", "code", "TS_CODE"])
    if ts_col is None:
        out["ts_code"] = ""
    else:
        out["ts_code"] = out[ts_col].astype(str)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _norm_ts_code(x: Union[str, float, int, None]) -> str:
    """
    统一 ts_code / code：
    - 若包含 .SZ/.SH 保留
    - 若纯6位数字：不强行补交易所（避免误判），但保留原值
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return s


def _strip_exchange(ts: str) -> str:
    """
    000001.SZ -> 000001
    000001.SH -> 000001
    """
    if not ts:
        return ""
    s = str(ts).strip()
    if "." in s:
        return s.split(".", 1)[0]
    return s


def _split_tags(v: Union[str, float, int, None]) -> List[str]:
    """
    支持 '机器人;AI,算力/军工' 等多分隔符。
    返回去空、去重后的 tag 列表（保持顺序大致稳定）。
    """
    if v is None:
        return []
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return []

    # 统一分隔符
    for sep in ["；", ";", "，", ",", "/", "|", "、", "\n", "\t"]:
        s = s.replace(sep, ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    # 去重
    out = []
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -------------------------
# Step0 context fetch (boards / dragon)
# -------------------------
def _get_step0_boards(s) -> pd.DataFrame:
    """
    兼容：s 可能是 Settings，也可能是 ctx(dict)
    期望得到 hot_boards DataFrame（包含 板块名 + 热度排名/热度值）
    也兼容包含 ts_code 的“股票->板块/概念”映射表
    """
    if s is None:
        return pd.DataFrame()

    if isinstance(s, dict):
        for k in ["boards", "hot_boards", "hot_board_df"]:
            v = s.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame()

    for attr in ["ctx", "context"]:
        if hasattr(s, attr):
            ctx = getattr(s, attr)
            if isinstance(ctx, dict):
                for k in ["boards", "hot_boards", "hot_board_df"]:
                    v = ctx.get(k, None)
                    if isinstance(v, pd.DataFrame):
                        return v
    return pd.DataFrame()


def _get_step0_dragon(s) -> pd.DataFrame:
    """
    兼容：龙虎榜/龙头数据
    """
    if s is None:
        return pd.DataFrame()

    if isinstance(s, dict):
        for k in ["dragon", "top_list", "dragon_df"]:
            v = s.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame()

    for attr in ["ctx", "context"]:
        if hasattr(s, attr):
            ctx = getattr(s, attr)
            if isinstance(ctx, dict):
                for k in ["dragon", "top_list", "dragon_df"]:
                    v = ctx.get(k, None)
                    if isinstance(v, pd.DataFrame):
                        return v
    return pd.DataFrame()


# -------------------------
# Build board rank map
# -------------------------
def _build_board_rank_map(boards: pd.DataFrame) -> pd.DataFrame:
    """
    返回标准化 DataFrame: [_board_name, _hot_rank]
    支持两种 boards：
      A) board_name + (hot_rank 或 hot)
      B) ts_code + board_name + (hot_rank 或 hot)
    """
    if boards is None or boards.empty:
        return pd.DataFrame(columns=["_board_name", "_hot_rank"])

    b = boards.copy()

    # 板块名字段
    b_name_col = _first_existing_col(b, ["board", "name", "concept", "theme", "industry", "板块", "概念"])
    if b_name_col is None:
        return pd.DataFrame(columns=["_board_name", "_hot_rank"])

    # 热度排名字段（没有就用热度值反推）
    b_rank_col = _first_existing_col(b, ["hot_rank", "rank", "board_hot_rank", "热度排名", "排名"])
    b_hot_col = _first_existing_col(b, ["hot", "heat", "score", "热度", "热度值"])

    b["_board_name"] = b[b_name_col].astype(str)

    if b_rank_col is not None:
        b["_hot_rank"] = pd.to_numeric(b[b_rank_col], errors="coerce")
    elif b_hot_col is not None:
        hot = pd.to_numeric(b[b_hot_col], errors="coerce").fillna(0.0)
        b["_hot_rank"] = hot.rank(ascending=False, method="min")
    else:
        b["_hot_rank"] = np.nan

    b = b[["_board_name", "_hot_rank"]].dropna()
    if b.empty:
        return pd.DataFrame(columns=["_board_name", "_hot_rank"])

    # 同名板块可能重复：取最热（最小rank）
    b["_hot_rank"] = pd.to_numeric(b["_hot_rank"], errors="coerce")
    b = b.dropna(subset=["_hot_rank"])
    if b.empty:
        return pd.DataFrame(columns=["_board_name", "_hot_rank"])

    b = b.groupby("_board_name", as_index=False)["_hot_rank"].min()
    return b


def _build_stock_to_board_map(boards: pd.DataFrame) -> pd.DataFrame:
    """
    若 boards 里包含 ts_code & board_name，则构建：[_ts, _board_name]
    """
    if boards is None or boards.empty:
        return pd.DataFrame(columns=["_ts", "_board_name"])

    b = boards.copy()
    ts_col = _first_existing_col(b, ["ts_code", "code", "TS_CODE"])
    b_name_col = _first_existing_col(b, ["board", "name", "concept", "theme", "industry", "板块", "概念"])

    if ts_col is None or b_name_col is None:
        return pd.DataFrame(columns=["_ts", "_board_name"])

    b["_ts"] = b[ts_col].astype(str).map(_norm_ts_code)
    b["_board_name"] = b[b_name_col].astype(str)
    b = b[["_ts", "_board_name"]].dropna()
    b = b[(b["_ts"] != "") & (b["_board_name"] != "")]
    if b.empty:
        return pd.DataFrame(columns=["_ts", "_board_name"])

    # 去重
    b = b.drop_duplicates()
    return b


# -------------------------
# Theme Boost Core
# -------------------------
def calc_theme_boost(strength_df: pd.DataFrame, s=None) -> pd.DataFrame:
    """
    ThemeBoost = 0.8 ~ 1.3 之间的平滑加权（sigmoid）
    - 板块热度（rank越小越热） -> _score_board
    - 龙虎榜/龙头奖励 -> _score_dragon
    """
    if strength_df is None or len(strength_df) == 0:
        return pd.DataFrame(columns=["ThemeBoost"])

    out = _ensure_ts_code(strength_df)

    # 规范化 ts_code（保留原字段同时给辅助字段）
    out["ts_code"] = out["ts_code"].astype(str).map(_norm_ts_code)
    out["_ts_plain"] = out["ts_code"].map(_strip_exchange)

    # 默认不热：50名
    out["board_hot_rank"] = pd.Series([50.0] * len(out), index=out.index, dtype="float64")

    # --- 1) 合并板块热度 ---
    boards = _get_step0_boards(s)

    # 1.1 构建 “板块名 -> rank” 映射
    board_rank_map = _build_board_rank_map(boards)

    # 1.2 决定：从 strength_df 自带板块字段，还是从 boards 里 ts_code 映射出板块
    stock_board_col = _first_existing_col(out, ["board", "concept", "theme", "industry", "板块", "概念"])

    if stock_board_col is not None and (not board_rank_map.empty):
        # strength_df 自带板块/概念字段：支持多题材拆分，取最热rank
        tmp = out[["ts_code", "_ts_plain", stock_board_col]].copy()
        tmp["_tags"] = tmp[stock_board_col].apply(_split_tags)
        tmp = tmp.explode("_tags")
        tmp["_board_name"] = tmp["_tags"].astype(str)
        tmp = tmp.dropna(subset=["_board_name"])
        tmp = tmp[tmp["_board_name"].astype(str).str.len() > 0]
        tmp = tmp.merge(board_rank_map, how="left", on="_board_name")

        # 每只股票取最小rank（最热）
        tmp["_hot_rank"] = pd.to_numeric(tmp["_hot_rank"], errors="coerce")
        best = (
            tmp.groupby("ts_code", as_index=False)["_hot_rank"]
            .min()
            .rename(columns={"_hot_rank": "_best_rank"})
        )
        out = out.merge(best, how="left", on="ts_code")
        out["board_hot_rank"] = pd.to_numeric(out["_best_rank"], errors="coerce").fillna(out["board_hot_rank"])
        out.drop(columns=[c for c in ["_best_rank"] if c in out.columns], inplace=True)

    else:
        # strength_df 没有板块字段：尝试 boards 提供 ts_code->板块 映射
        stock_to_board = _build_stock_to_board_map(boards)
        if (not stock_to_board.empty) and (not board_rank_map.empty):
            # out ts_code 与映射表对齐（兼容 000001 与 000001.SZ）
            stb = stock_to_board.copy()
            stb["_ts_plain"] = stb["_ts"].map(_strip_exchange)

            # 用 plain code 先对齐
            m = out[["ts_code", "_ts_plain"]].merge(
                stb[["_ts_plain", "_board_name"]],
                how="left",
                on="_ts_plain",
            )
            m = m.merge(board_rank_map, how="left", on="_board_name")

            # 每只股票取最热rank
            m["_hot_rank"] = pd.to_numeric(m["_hot_rank"], errors="coerce")
            best = (
                m.groupby("ts_code", as_index=False)["_hot_rank"]
                .min()
                .rename(columns={"_hot_rank": "_best_rank"})
            )
            out = out.merge(best, how="left", on="ts_code")
            out["board_hot_rank"] = pd.to_numeric(out["_best_rank"], errors="coerce").fillna(out["board_hot_rank"])
            out.drop(columns=[c for c in ["_best_rank"] if c in out.columns], inplace=True)

    out["board_hot_rank"] = pd.to_numeric(out["board_hot_rank"], errors="coerce").fillna(50.0).astype("float64")
    out["board_hot_rank"] = out["board_hot_rank"].clip(1.0, 200.0)

    # --- 2) 龙虎榜/龙头奖励 ---
    dragon = _get_step0_dragon(s)
    dragon_flag = pd.Series([0.0] * len(out), index=out.index, dtype="float64")

    if dragon is not None and (not dragon.empty):
        ts_col = _first_existing_col(dragon, ["ts_code", "code", "TS_CODE"])
        if ts_col is not None:
            d = dragon.copy()
            d["_ts"] = d[ts_col].astype(str).map(_norm_ts_code)
            d["_ts_plain"] = d["_ts"].map(_strip_exchange)
            dragon_set_plain = set(d["_ts_plain"].astype(str).tolist())
            dragon_flag = out["_ts_plain"].astype(str).apply(lambda x: 1.0 if x in dragon_set_plain else 0.0).astype("float64")

    out["dragon_flag"] = dragon_flag.clip(0.0, 1.0)

    # --- 3) 分项打分 ---
    # rank 越小越热：1名接近1分，50名接近0分
    out["_score_board"] = (1.0 - (out["board_hot_rank"] / 50.0)).clip(0.0, 1.0)
    out["_score_dragon"] = out["dragon_flag"].clip(0.0, 1.0)

    # --- 4) Sigmoid 合成（0.8~1.3） ---
    raw = out["_score_board"].astype(float).values + 0.6 * out["_score_dragon"].astype(float).values
    out["ThemeBoost"] = 0.8 + 0.5 * _sigmoid(3.0 * (raw - 0.5))

    # 清理内部辅助列（保留可解释字段）
    for c in ["_ts_plain"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True)

    return out.sort_values("ThemeBoost", ascending=False)


# -------------------------
# Runner
# -------------------------
def run_step4(strength_df: pd.DataFrame, s=None) -> pd.DataFrame:
    return calc_theme_boost(strength_df, s=s)


# Backward-compatible alias
def run(df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step4(df, s=s)


if __name__ == "__main__":
    print("Step4 ThemeBoost ready.")
