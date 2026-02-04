from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """返回 df 中第一个命中的列名（大小写不敏感）。"""
    if df is None or df.empty:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit:
            return hit
    return None


def _ensure_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    兼容不同快照字段命名：
    - ts_code 可能叫 ts_code / code / 股票代码 / TS_CODE ...
    - name 可能叫 name / ts_name / 股票名称 / NAME ...
    """
    if df is None:
        df = pd.DataFrame()

    df = df.copy()

    ts_col = _first_existing_col(df, ["ts_code", "code", "股票代码", "证券代码", "TS_CODE"])
    name_col = _first_existing_col(df, ["name", "ts_name", "股票名称", "证券简称", "NAME", "TS_NAME"])

    if ts_col is None:
        # 实在找不到就创建空列，保证流程不炸
        df["ts_code"] = ""
        ts_col = "ts_code"
    elif ts_col != "ts_code":
        df.rename(columns={ts_col: "ts_code"}, inplace=True)
        ts_col = "ts_code"

    if name_col is None:
        df["name"] = ""
        name_col = "name"
    elif name_col != "name":
        df.rename(columns={name_col: "name"}, inplace=True)
        name_col = "name"

    return df, ts_col, name_col


def _is_bad_name(x: Any) -> bool:
    """
    H1 粗过滤：剔除 ST / *ST / 退 / 退市
    说明：这里是“最小闭环”的粗过滤；如果后续快照有 is_st / delist 字段，可再增强。
    """
    t = str(x or "")
    # 注意：先判断 *ST/ ST，再判断退市相关
    return ("*ST" in t) or ("ST" in t) or ("退市" in t) or ("退" in t)


def _is_limit_close_sealed(df: pd.DataFrame) -> pd.Series:
    """
    判断“收盘封死涨停”的尽力版本（兼容不同字段）：
    - 优先使用：close==high 且 (可选) open < close
    - 若存在 is_seal / sealed / 封单 等字段则优先（但目前不强依赖）
    - 若存在 limit_up 或 up_limit，则 close==limit_up 也可认为封死（但仍以 close==high 为主）
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=bool)

    close_col = _first_existing_col(df, ["close", "收盘", "CLOSE"])
    high_col = _first_existing_col(df, ["high", "最高", "HIGH"])
    open_col = _first_existing_col(df, ["open", "开盘", "OPEN"])
    limit_up_col = _first_existing_col(df, ["limit_up", "up_limit", "涨停价", "UP_LIMIT"])

    # 兜底：没有价格字段，就只能“全 True”（让上游 limit_list_d 自己保证是涨停池）
    if close_col is None or high_col is None:
        return pd.Series([True] * n, index=df.index)

    close = pd.to_numeric(df[close_col], errors="coerce")
    high = pd.to_numeric(df[high_col], errors="coerce")

    sealed = (close.notna()) & (high.notna()) & (close == high)

    # 可选增强：如果有涨停价字段，则进一步要求 close==limit_up（但不强制，避免数据源不一致）
    if limit_up_col is not None:
        limit_up = pd.to_numeric(df[limit_up_col], errors="coerce")
        # 只在 limit_up 非空的行启用该约束
        sealed = sealed & ((limit_up.isna()) | (close == limit_up))

    # 可选增强：如果有开盘价，避免极端脏数据（开盘=收盘=最高也仍可通过）
    if open_col is not None:
        op = pd.to_numeric(df[open_col], errors="coerce")
        sealed = sealed & (op.isna() | (op <= close))

    return sealed.fillna(False)


def step2_build_candidates(s: Settings, ctx: Dict[str, Any]) -> pd.DataFrame:
    """
    冻结口径 Step2（硬条件筛选）：
    - 今日涨停（收盘封死）
    - 非ST非退市风险（H1） —— 先用名称字段粗过滤（含ST/退）做最小闭环
    - 非极端一字板（可选） —— V0.2 不做（缺分钟/盘口数据）

    输出：CandidateSet（30~200只股票，实际取决于当日涨停数量）
    """
    raw: Any = ctx.get("limit_list_d", pd.DataFrame())
    limit_df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)

    if limit_df is None or limit_df.empty:
        return pd.DataFrame(columns=["ts_code", "name"])

    limit_df, ts_col, name_col = _ensure_columns(limit_df)

    # 1) 先确保“收盘封死涨停”
    sealed_mask = _is_limit_close_sealed(limit_df)

    # 2) H1 粗过滤：剔除 ST / *ST / 退 / 退市
    bad_mask = limit_df[name_col].apply(_is_bad_name)

    cand = (
        limit_df.loc[sealed_mask & (~bad_mask), [ts_col, name_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 兜底：保证列顺序与下游一致
    cand = cand.rename(columns={ts_col: "ts_code", name_col: "name"})[["ts_code", "name"]]

    return cand
