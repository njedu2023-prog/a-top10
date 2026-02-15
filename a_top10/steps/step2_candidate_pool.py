from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, List

import pandas as pd

from a_top10.config import Settings


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """返回 df 中第一个命中的列名（大小写不敏感）。"""
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    lower_map = {c.lower(): c0 for c, c0 in zip(cols, df.columns)}
    for c in candidates:
        hit = lower_map.get(str(c).lower())
        if hit:
            return hit
    return None


def _normalize_ts_code_one(x: Any) -> str:
    """
    ts_code 统一成 000001.SZ / 600000.SH / 830xxx.BJ（大写）
    兼容输入：
      - 000001
      - 000001.SZ / 000001sz / 000001.sz
      - 600000.SH
    """
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    s = s.replace(" ", "").upper()

    # 已带后缀
    if "." in s:
        left, right = s.split(".", 1)
        left = "".join([c for c in left if c.isdigit()])
        right = "".join([c for c in right if c.isalpha()])
        if len(left) == 6 and right in {"SZ", "SH", "BJ"}:
            return f"{left}.{right}"
        if len(left) == 6:
            s6 = left
            if s6.startswith(("6", "9")):
                return f"{s6}.SH"
            if s6.startswith(("8", "4")):
                return f"{s6}.BJ"
            return f"{s6}.SZ"
        return ""

    # 无后缀：提取 6 位数字
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) != 6:
        return ""

    if digits.startswith(("6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("8", "4")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _normalize_ts_code_series(sr: pd.Series) -> pd.Series:
    if sr is None:
        return pd.Series([], dtype=str)
    return sr.fillna("").astype(str).map(_normalize_ts_code_one)


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

    # ✅ 关键：归一化 ts_code，避免 merge 对不上
    df["ts_code"] = _normalize_ts_code_series(df["ts_code"])

    return df, ts_col, name_col


def _ensure_ts_code(df: pd.DataFrame) -> pd.DataFrame:
    """确保任意表都有 ts_code 列（用于 merge），并归一化。"""
    if df is None:
        return pd.DataFrame(columns=["ts_code"])
    df = df.copy()

    if df.empty:
        if "ts_code" not in df.columns:
            df["ts_code"] = ""
        else:
            df["ts_code"] = _normalize_ts_code_series(df["ts_code"])
        return df

    ts_col = _first_existing_col(df, ["ts_code", "code", "股票代码", "证券代码", "TS_CODE"])
    if ts_col is None:
        df["ts_code"] = ""
    elif ts_col != "ts_code":
        df.rename(columns={ts_col: "ts_code"}, inplace=True)

    # ✅ 关键：归一化 ts_code
    df["ts_code"] = _normalize_ts_code_series(df["ts_code"])
    return df


def _get_df_from_ctx(ctx: Dict[str, Any], keys: List[str]) -> pd.DataFrame:
    """
    从 ctx 里按多个 key 尝试取 DataFrame。
    兼容：ctx 可能用文件名、也可能用不带扩展名的 key。
    """
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
        # 有些 loader 可能塞进 list[dict]
        if v is not None and not isinstance(v, pd.DataFrame):
            try:
                df = pd.DataFrame(v)
                if not df.empty:
                    return df
            except Exception:
                pass
    return pd.DataFrame()


def _is_bad_name(x: Any) -> bool:
    """
    H1 粗过滤：剔除 ST / *ST / 退 / 退市
    说明：这里是“最小闭环”的粗过滤；如果后续快照有 is_st / delist 字段，可再增强。
    """
    t = str(x or "")
    return ("*ST" in t) or ("ST" in t) or ("退市" in t) or ("退" in t)


def _is_limit_close_sealed(df: pd.DataFrame) -> pd.Series:
    """
    判断“收盘封死涨停”的尽力版本（兼容不同字段）：
    - 优先使用：close==high 且 (可选) open < close
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

    if limit_up_col is not None:
        limit_up = pd.to_numeric(df[limit_up_col], errors="coerce")
        sealed = sealed & ((limit_up.isna()) | (close == limit_up))

    if open_col is not None:
        op = pd.to_numeric(df[open_col], errors="coerce")
        sealed = sealed & (op.isna() | (op <= close))

    return sealed.fillna(False)


def _select_rename(df: pd.DataFrame, mapping: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """
    从 df 中挑选 mapping 里需要的字段，并统一 rename 成标准字段名。
    mapping: {标准名: [候选列名1,2,3...]}
    返回至少包含 ts_code + 命中的标准列
    """
    df = _ensure_ts_code(df)
    if df.empty:
        return pd.DataFrame(columns=["ts_code"] + list(mapping.keys()))

    out = df[["ts_code"]].copy()
    for std_name, candidates in mapping.items():
        col = _first_existing_col(df, list(candidates))
        if col is None:
            continue
        out[std_name] = df[col]
    return out


def _merge_left(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if base is None or base.empty:
        return base
    if extra is None or extra.empty:
        return base
    extra = _ensure_ts_code(extra)
    if "ts_code" not in extra.columns:
        return base

    # 防止重复列覆盖：仅追加 base 没有的列
    add_cols = [c for c in extra.columns if c != "ts_code" and c not in base.columns]
    if not add_cols:
        return base
    return base.merge(extra[["ts_code"] + add_cols], on="ts_code", how="left")


def step2_build_candidates(s: Settings, ctx: Dict[str, Any]) -> pd.DataFrame:
    """
    冻结口径 Step2（硬条件筛选）：
    - 今日涨停（收盘封死）
    - 非ST非退市风险（H1） —— 先用名称字段粗过滤（含ST/退）做最小闭环
    - 非极端一字板（可选） —— V0.2 不做（缺分钟/盘口数据）

    输出：CandidateSet（30~200只股票，实际取决于当日涨停数量）
    并在此处“尽力补齐 Step3 需要的字段”（有啥给啥，不强依赖）
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
        limit_df.loc[sealed_mask & (~bad_mask), ["ts_code", "name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ✅ 再保险：候选集合 ts_code 也归一化（避免上游未规范）
    cand["ts_code"] = _normalize_ts_code_series(cand["ts_code"])

    # -----------------------------
    # ✅ 关键：补齐 Step3 可能用到的字段
    # -----------------------------

    # 2.1 daily：涨跌幅/成交额/换手率（不同数据源列名不一致，尽量映射）
    daily_df = _get_df_from_ctx(ctx, ["daily", "daily.csv"])
    daily_sel = _select_rename(
        daily_df,
        mapping={
            "pct_change": ["pct_change", "change_pct", "pct_chg", "涨跌幅", "PCT_CHG"],
            "amount": ["amount", "成交额", "turnover_amount", "amt", "AMOUNT"],
            "turnover_rate": [
                "turnover_rate",
                "turn_rate",
                "换手率",
                "turnover",
                "TURNOVER_RATE",
                "TURN_RATE",
            ],
        },
    )
    cand = _merge_left(cand, daily_sel)

    # 2.2 daily_basic：流通市值（float_values）
    daily_basic_df = _get_df_from_ctx(ctx, ["daily_basic", "daily_basic.csv"])
    basic_sel = _select_rename(
        daily_basic_df,
        mapping={
            "float_values": ["float_values", "float_mv", "流通市值", "FLOAT_MV"],
        },
    )
    cand = _merge_left(cand, basic_sel)

    # 2.3 moneyflow_hsgt（或 moneyflow）：净流入/净占比（若有就给）
    mf_df = _get_df_from_ctx(ctx, ["moneyflow_hsgt", "moneyflow_hsgt.csv", "moneyflow", "moneyflow.csv"])
    mf_sel = _select_rename(
        mf_df,
        mapping={
            "net_amount": ["net_amount", "net_amt", "净额", "净买入额", "NET_AMOUNT", "NET_AMT"],
            "net_rate": ["net_rate", "净占比", "net_ratio", "NET_RATE", "NET_RATIO"],
        },
    )
    cand = _merge_left(cand, mf_sel)

    # 2.4 top_list（龙虎榜）：龙虎榜成交额 / 成交额占比（若有就给）
    top_df = _get_df_from_ctx(ctx, ["top_list", "top_list.csv", "lhb", "lhb.csv"])
    top_sel = _select_rename(
        top_df,
        mapping={
            "l_amount": ["l_amount", "龙虎榜成交额", "lhb_amount", "amount", "成交额", "AMOUNT"],
            "amount_rate": ["amount_rate", "成交额占比", "lhb_amount_rate", "rate", "比例", "AMOUNT_RATE"],
        },
    )
    cand = _merge_left(cand, top_sel)

    # 兜底：保证列顺序（前两列固定）
    fixed = ["ts_code", "name"]
    rest = [c for c in cand.columns if c not in fixed]
    cand = cand[fixed + rest]

    return cand
