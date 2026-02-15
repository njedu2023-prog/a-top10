#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step2：Candidate Pool（终版，工程对齐）
- 主程序依赖：from a_top10.steps.step2_candidate_pool import step2_build_candidates
- 关键修复/增强：
  1) ✅ 统一 ts_code 列（兼容 ts_code/code/symbol/证券代码 等）
  2) ✅ 强制补齐 industry（来自 stock_basic.csv / ctx["stock_basic"]）
  3) ✅ 输出旁路 debug：outputs/debug_step2_candidate_YYYYMMDD.json
  4) ✅ 返回值类型对齐：必须返回 DataFrame（避免 Step3 收到 dict 崩溃）
  5) ✅ 更强健：ctx 为空/脏数据/异常写盘都不崩（尽量落 debug）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


# =========================
# Utils
# =========================

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # 最差情况下也不要影响主流程
        pass


def _json_fallback(o: Any) -> Any:
    """json.dumps default fallback"""
    try:
        return str(o)
    except Exception:
        return "<unserializable>"


def _safe_json_dump(obj: Any, path: Path) -> None:
    try:
        _ensure_dir(path.parent)
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_json_fallback),
            encoding="utf-8",
        )
    except Exception:
        # debug 写不进去也不要影响主流程
        pass


def _first_existing_col(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_str_series(df: Optional[pd.DataFrame], col: Optional[str]) -> pd.Series:
    if df is None:
        return pd.Series([], dtype="string")
    if df.empty or not col or col not in df.columns:
        return pd.Series([""] * len(df), dtype="string")
    # 注意：astype("string") 可能把非标对象变为 <NA>，这里统一 strip
    return df[col].astype("string").fillna("").map(lambda x: str(x).strip())


def _normalize_ts_code_value(x: str) -> str:
    """
    兼容：
    - 000001.SZ
    - SZ000001 / SH600000 / BJxxxxxx
    - 000001SZ / 600000SH
    - 000001（无法判断交易所则原样返回）
    """
    s = (x or "").strip().upper()
    if not s:
        return s

    # 000001.SZ
    if "." in s:
        parts = s.split(".", 1)
        code = parts[0].strip()
        exch = parts[1].strip().replace("SSE", "SH").replace("SZSE", "SZ")
        if len(code) == 6 and exch in ("SH", "SZ", "BJ"):
            return f"{code}.{exch}"
        return s

    # SZ000001 / SH600000 / BJxxxxxx
    if s.startswith(("SZ", "SH", "BJ")) and len(s) >= 8:
        exch = s[:2]
        code = s[2:8]
        if code.isdigit():
            return f"{code}.{exch}"

    # 000001SZ / 600000SH
    if len(s) >= 8 and s[:6].isdigit() and s[6:8] in ("SZ", "SH", "BJ"):
        return f"{s[:6]}.{s[6:8]}"

    return s


def _normalize_id_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    把各种代码列统一为 ts_code，并规范化值。
    """
    dbg: Dict[str, Any] = {"found_code_col": "", "normalized_ratio": 0.0}

    if df is None or df.empty:
        return df, dbg

    code_col = _first_existing_col(
        df,
        [
            "ts_code", "TS_CODE",
            "code", "CODE",
            "symbol", "SYMBOL",
            "证券代码", "代码", "股票代码",
        ],
    )
    dbg["found_code_col"] = code_col or ""
    if not code_col:
        return df, dbg

    out = df.copy()
    out["ts_code"] = _to_str_series(out, code_col).map(_normalize_ts_code_value)
    dbg["normalized_ratio"] = float((out["ts_code"].astype("string").fillna("") != "").mean())
    return out, dbg


def _resolve_outputs_dir(s: Settings) -> Path:
    # 兼容不同 Settings 字段命名
    for key in ["outputs_dir", "output_dir", "outputs", "out_dir"]:
        if hasattr(s, key):
            v = getattr(s, key)
            if v is None:
                continue
            try:
                return Path(v)
            except Exception:
                continue
    return Path("outputs")


def _pick_trade_date(s: Settings, ctx: Dict[str, Any]) -> str:
    td = str(ctx.get("trade_date", "") or "").strip()
    if td:
        return td
    if hasattr(s, "trade_date") and getattr(s, "trade_date"):
        return str(getattr(s, "trade_date")).strip()
    return "unknown"


def _ctx_get_df(ctx: Dict[str, Any], keys: List[str]) -> Optional[pd.DataFrame]:
    """
    只接受真正的 DataFrame，避免把 dict/list 误传下去。
    """
    for k in keys:
        v = ctx.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


# =========================
# Step2 main
# =========================

def step2_build_candidates(s: Settings, ctx: Dict[str, Any]) -> pd.DataFrame:
    """
    ✅ 返回：candidates_df（DataFrame）
    同时把 candidates_df 写回 ctx，供后续步骤取用：
      ctx["candidates"] = candidates_df
      ctx["step2"] = candidates_df   # 兼容旧链路
    """
    if ctx is None:
        ctx = {}

    trade_date = _pick_trade_date(s, ctx)
    out_dir = _resolve_outputs_dir(s)
    _ensure_dir(out_dir)

    debug: Dict[str, Any] = {
        "trade_date": trade_date,
        "base_source": "",
        "base_rows": 0,
        "code_norm": {},
        "stock_basic_rows": 0,
        "industry_merge": {
            "ok": False,
            "reason": "",
            "industry_nonblank_ratio_before": 0.0,
            "industry_nonblank_ratio_after": 0.0,
        },
        "filters": {"drop_st": 0, "drop_delist": 0},
        "final_rows": 0,
        "final_cols": [],
        "debug_file": str(out_dir / f"debug_step2_candidate_{trade_date}.json"),
        "out_csv": str(out_dir / f"step2_candidates_{trade_date}.csv"),
    }

    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        # 兜底：保证返回 DataFrame
        if df is None or not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        debug["final_rows"] = int(len(df))
        debug["final_cols"] = list(df.columns)

        # 尽量写盘（失败不影响主流程）
        try:
            df.to_csv(Path(debug["out_csv"]), index=False, encoding="utf-8-sig")
        except Exception:
            pass

        _safe_json_dump(debug, Path(debug["debug_file"]))

        # 写回 ctx（始终写 DataFrame）
        ctx["candidates"] = df
        ctx["step2"] = df
        return df

    # 1) base：优先涨停列表
    base_df = _ctx_get_df(ctx, ["limit_list_d", "stk_limit", "limit_up", "limit_up_list"])
    if base_df is not None and not base_df.empty:
        debug["base_source"] = "limit_list_d/stk_limit"
    else:
        base_df = _ctx_get_df(ctx, ["top_list", "daily", "daily_basic"])
        debug["base_source"] = "top_list/daily/daily_basic" if base_df is not None else ""

    if base_df is None or base_df.empty:
        return _finalize(pd.DataFrame())

    base_df = base_df.copy()
    debug["base_rows"] = int(len(base_df))

    # 2) 统一 ts_code
    base_df, code_dbg = _normalize_id_columns(base_df)
    debug["code_norm"] = code_dbg

    if "ts_code" not in base_df.columns:
        # 没有代码列就不继续做 merge/去重（但仍保证返回 df）
        return _finalize(base_df)

    # 3) 补 industry
    before_ratio = 0.0
    if "industry" in base_df.columns:
        before_ratio = float((base_df["industry"].astype("string").fillna("") != "").mean())
    debug["industry_merge"]["industry_nonblank_ratio_before"] = before_ratio

    sb = _ctx_get_df(ctx, ["stock_basic", "stock_basic_df"])
    if sb is None or sb.empty:
        debug["industry_merge"]["ok"] = False
        debug["industry_merge"]["reason"] = "stock_basic missing in ctx"
        candidates_df = base_df
    else:
        sb = sb.copy()
        debug["stock_basic_rows"] = int(len(sb))

        sb, sb_code_dbg = _normalize_id_columns(sb)

        # stock_basic 必须也有 ts_code 才能 merge
        if "ts_code" not in sb.columns:
            debug["industry_merge"]["ok"] = False
            debug["industry_merge"]["reason"] = "ts_code missing in stock_basic after normalization"
            candidates_df = base_df
        else:
            sb_ind_col = _first_existing_col(sb, ["industry", "行业", "industry_name"])
            if not sb_ind_col:
                debug["industry_merge"]["ok"] = False
                debug["industry_merge"]["reason"] = "industry col missing in stock_basic"
                candidates_df = base_df
            else:
                sb2 = sb[["ts_code", sb_ind_col]].rename(columns={sb_ind_col: "industry"}).copy()
                sb2["industry"] = sb2["industry"].astype("string").fillna("").map(lambda x: str(x).strip())

                candidates_df = base_df.merge(sb2, on="ts_code", how="left", suffixes=("", "_sb"))
                # 理论上 sb2 已经叫 industry，不会有 industry_sb；这里保留兼容逻辑
                if "industry_sb" in candidates_df.columns:
                    if "industry" in candidates_df.columns:
                        base_ind = candidates_df["industry"].astype("string").fillna("")
                        sb_ind = candidates_df["industry_sb"].astype("string").fillna("")
                        candidates_df["industry"] = base_ind.where(base_ind != "", sb_ind)
                    else:
                        candidates_df["industry"] = candidates_df["industry_sb"]
                    candidates_df.drop(columns=["industry_sb"], inplace=True, errors="ignore")

                debug["industry_merge"]["ok"] = True
                debug["industry_merge"]["reason"] = ""
                debug["industry_merge"]["stock_basic_code_norm"] = sb_code_dbg

    after_ratio = 0.0
    if candidates_df is not None and (not candidates_df.empty) and "industry" in candidates_df.columns:
        after_ratio = float((candidates_df["industry"].astype("string").fillna("") != "").mean())
    debug["industry_merge"]["industry_nonblank_ratio_after"] = after_ratio

    # 4) 过滤 ST / 退（可选）
    name_col = _first_existing_col(candidates_df, ["name", "名称", "股票简称", "ts_name"])
    if name_col:
        name_s = _to_str_series(candidates_df, name_col)
        mask_st = name_s.str.contains("ST", case=False, regex=False)
        debug["filters"]["drop_st"] = int(mask_st.sum())
        candidates_df = candidates_df.loc[~mask_st].copy()

        name_s2 = _to_str_series(candidates_df, name_col)
        mask_delist = name_s2.str.contains("退", regex=False) | name_s2.str.contains("退市", regex=False)
        debug["filters"]["drop_delist"] = int(mask_delist.sum())
        candidates_df = candidates_df.loc[~mask_delist].copy()

    # 5) 去重 & 列前置
    candidates_df["ts_code"] = (
        candidates_df["ts_code"]
        .astype("string")
        .fillna("")
        .map(_normalize_ts_code_value)
    )
    candidates_df = candidates_df.loc[candidates_df["ts_code"] != ""].copy()
    candidates_df = candidates_df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)

    front = [c for c in ["ts_code", "name", "industry"] if c in candidates_df.columns]
    rest = [c for c in candidates_df.columns if c not in front]
    candidates_df = candidates_df[front + rest]

    return _finalize(candidates_df)


# 兼容旧调用（如果你工程里有人用 step2_build_candidates.run 之类）
def run(df: Any, s: Optional[Settings] = None, ctx: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    兼容入口：允许直接给 df（作为 base）来构建 candidates。
    - 如果传 df：会写入 ctx["limit_list_d"]，优先作为 base_source
    """
    if ctx is None:
        ctx = {}
    if isinstance(df, pd.DataFrame):
        ctx.setdefault("limit_list_d", df)

    if s is None:
        # 兜底：如果外部没给 Settings，尽量构造一个
        try:
            s = Settings()
        except Exception:
            # 极端情况下用一个空壳，outputs_dir 等会走默认
            class _S:  # type: ignore
                pass
            s = _S()  # type: ignore

    return step2_build_candidates(s, ctx)


if __name__ == "__main__":
    print("Step2 (Candidate Pool) module loaded successfully.")
