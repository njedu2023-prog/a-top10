from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from a_top10.config import Settings


# -------------------------
# IO Utils
# -------------------------
def _read_csv_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    # 依次尝试多种编码（快照经常是 utf-8-sig / gbk）
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def _dir_has_any_csv(d: Path, filenames: List[str]) -> bool:
    if not d.exists() or (not d.is_dir()):
        return False
    for fn in filenames:
        if (d / fn).exists():
            return True
    return False


def _pick_snapshot_dir(s: Settings, trade_date: str) -> Tuple[Path, List[str]]:
    """
    选择一个最可能的快照目录：
    - 优先：s.data_repo.snapshot_dir(trade_date)
    - 兜底：尝试一组常见目录（不写死具体仓库名，只写“形态”）
    返回：
      chosen_dir, tried_dirs(str list)

    ✅ 关键修复：不能只看目录存在，必须至少命中关键 CSV，否则会选到空目录导致下游全空。
    """
    tried: List[str] = []

    key_files = [
        "daily.csv",
        "daily_basic.csv",
        "limit_list_d.csv",
        "limit_break_d.csv",
        "hot_boards.csv",
        "top_list.csv",
        "stock_basic.csv",
    ]

    # 1) 主推荐路径
    primary = s.data_repo.snapshot_dir(trade_date)
    tried.append(str(primary))
    if primary.exists() and primary.is_dir() and _dir_has_any_csv(primary, key_files):
        return primary, tried

    # 2) 兜底目录形态
    year = trade_date[:4]

    repo_name = None
    try:
        repo_name = getattr(getattr(s, "data_repo", None), "name", None)
    except Exception:
        repo_name = None
    if not repo_name:
        repo_name = "a-share-top3-data"

    candidates: List[Path] = [
        Path("snap"),
        Path("snap") / trade_date,
        Path("snapshots") / trade_date,
        Path("data_repo") / "snapshots" / trade_date,
        Path("_warehouse") / repo_name / "data" / "raw" / year / trade_date,
        Path("_warehouse") / repo_name / "raw" / year / trade_date,
        Path("_warehouse") / "data" / "raw" / year / trade_date,
    ]

    for d in candidates:
        tried.append(str(d))
        if _dir_has_any_csv(d, key_files):
            return d, tried

    return primary, tried


def _build_missing_list(snapshot_dir: Path, files: List[str]) -> List[str]:
    missing: List[str] = []
    for fn in files:
        if not (snapshot_dir / fn).exists():
            missing.append(fn)
    return missing


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        k = str(c).lower()
        if k in lower_map:
            return lower_map[k]
    return None


def _safe_str_series(sr: pd.Series) -> pd.Series:
    """
    统一清洗：
    - NaN -> ""
    - "nan"/"<na>"/"None"/"null" 等字符串 -> ""
    - strip
    """
    s = sr.fillna("").astype(str).str.strip()
    s = s.replace(
        {
            "nan": "",
            "NaN": "",
            "<NA>": "",
            "<na>": "",
            "None": "",
            "NONE": "",
            "null": "",
            "NULL": "",
        }
    )
    return s


def _normalize_industry_name(x: str) -> str:
    """行业名轻量归一：去空白、全角空格、常见分隔符收敛。"""
    if x is None:
        return ""
    s = str(x).strip().replace("\u3000", " ")
    s = " ".join(s.split())
    # 常见分隔符统一成空格（避免“汽车/配件”“汽车-配件”对不上）
    for sep in ["/", "\\", "|", "｜", "-", "－", "—", "_"]:
        s = s.replace(sep, " ")
    s = " ".join(s.split())
    return s


def _normalize_ts_code_one(x: str) -> str:
    """
    终极修复关键点：ts_code 统一成 000001.SZ / 600000.SH / 830xxx.BJ（大写）
    兼容输入：
      - 000001
      - 000001.SZ / 000001sz / 000001.sz
      - 600000.SH
    """
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    s = s.replace(" ", "").upper()

    # 已是类似 000001.SZ
    if "." in s:
        left, right = s.split(".", 1)
        left = "".join([c for c in left if c.isdigit()])
        right = "".join([c for c in right if c.isalpha()])
        if len(left) == 6 and right in {"SZ", "SH", "BJ"}:
            return f"{left}.{right}"
        # 其它带点奇形怪状，尽量提取 6 位数字
        if len(left) == 6:
            # 没法判断交易所就按规则补
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

    # 交易所推断（A股通用约定）
    if digits.startswith(("6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("8", "4")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _normalize_ts_code_series(sr: pd.Series) -> pd.Series:
    s = _safe_str_series(sr)
    return s.map(_normalize_ts_code_one)


def _normalize_industry_table(df: pd.DataFrame, *, table_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    把“行业/板块”统一规范为 df["industry"]，并保证 df["ts_code"] 存在且已归一。
    返回：(new_df, debug_info)
    """
    dbg: Dict[str, Any] = {
        "table": table_name,
        "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "has_ts_code": False,
        "code_col_used": None,
        "industry_col_used": None,
        "industry_nonempty": 0,
        "ts_code_nonempty": 0,
    }

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, dbg

    out = df.copy()

    # 1) code 列 -> ts_code（并归一）
    code_col = _first_existing_col(out, ["ts_code", "code", "股票代码", "证券代码", "TS_CODE"])
    if code_col:
        dbg["code_col_used"] = code_col
        if "ts_code" not in out.columns:
            out["ts_code"] = _normalize_ts_code_series(out[code_col])
        else:
            out["ts_code"] = _normalize_ts_code_series(out["ts_code"])
        dbg["has_ts_code"] = True
    else:
        if "ts_code" not in out.columns:
            out["ts_code"] = ""
        else:
            out["ts_code"] = _normalize_ts_code_series(out["ts_code"])

    # 2) 行业列 -> industry（并归一）
    ind_col = _first_existing_col(
        out,
        [
            "industry",
            "行业",
            "所属行业",
            "所属板块",
            "板块",
            "板块名称",
            "行业名称",
            "申万行业",
            "sw_industry",
            "sw_industry_name",
            "concept",
            "题材",
        ],
    )
    if ind_col:
        dbg["industry_col_used"] = ind_col
        out["industry"] = _safe_str_series(out[ind_col]).map(_normalize_industry_name)
    else:
        if "industry" not in out.columns:
            out["industry"] = ""
        else:
            out["industry"] = _safe_str_series(out["industry"]).map(_normalize_industry_name)

    # 3) 去重：同一 ts_code 保留第一条非空 industry（更稳）
    if "ts_code" in out.columns:
        out["ts_code"] = _normalize_ts_code_series(out["ts_code"])
        # 先让非空行业排前面
        out["_ind_nonempty"] = (out["industry"].astype(str).str.strip() != "").astype(int)
        out = out.sort_values(by=["ts_code", "_ind_nonempty"], ascending=[True, False])
        out = out.drop_duplicates(subset=["ts_code"], keep="first")
        out = out.drop(columns=["_ind_nonempty"], errors="ignore").reset_index(drop=True)

    dbg["industry_nonempty"] = int((out["industry"].astype(str).str.strip() != "").sum())
    dbg["ts_code_nonempty"] = int((out["ts_code"].astype(str).str.strip() != "").sum())
    return out, dbg


# -------------------------
# Step0 Core
# -------------------------
def step0_build_universe(s: Settings, trade_date: str) -> Dict[str, Any]:
    """
    Step0: Input Layer
    - 负责从“数据仓库快照目录”读入各类原始 CSV
    - 产出 ctx（后续 step1~step6 都从 ctx 取数据）
    - 输出关键 debug 字段：snapshot_dir / snapshot_dir_tried / snapshot_missing

    ✅ 行业链路终极修复目标：
    1) 所有表的 ts_code 统一成 000001.SZ/SH/BJ
    2) universe 一定包含行业列（同时提供 “行业” 与 “industry” 两个同义字段）
    """
    # 1) 选择快照目录
    snap, snap_tried = _pick_snapshot_dir(s, trade_date)

    # 2) 文件集合
    files_needed = [
        "daily.csv",
        "daily_basic.csv",
        "limit_list_d.csv",
        "limit_break_d.csv",
        "hot_boards.csv",
        "top_list.csv",
        "stock_basic.csv",
    ]
    snapshot_missing = _build_missing_list(snap, files_needed)

    # 3) 读取
    daily = _read_csv_if_exists(snap / "daily.csv")
    daily_basic_raw = _read_csv_if_exists(snap / "daily_basic.csv")
    limit_list_d = _read_csv_if_exists(snap / "limit_list_d.csv")
    limit_break_d = _read_csv_if_exists(snap / "limit_break_d.csv")
    hot_boards = _read_csv_if_exists(snap / "hot_boards.csv")
    top_list = _read_csv_if_exists(snap / "top_list.csv")
    stock_basic_raw = _read_csv_if_exists(snap / "stock_basic.csv")

    # 4) 规范化：stock_basic / daily_basic（行业链路主数据源）
    stock_basic, dbg_stock_basic = _normalize_industry_table(stock_basic_raw, table_name="stock_basic")
    daily_basic, dbg_daily_basic = _normalize_industry_table(daily_basic_raw, table_name="daily_basic")

    # 5) 规范化：daily / limit / top_list（至少统一 ts_code，避免 join 对不上）
    def _normalize_ts_code_inplace(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        ts_col = _first_existing_col(df, ["ts_code", "code", "TS_CODE", "股票代码", "证券代码"])
        if ts_col:
            df = df.copy()
            df["ts_code"] = _normalize_ts_code_series(df[ts_col])
        return df

    daily = _normalize_ts_code_inplace(daily)
    limit_list_d = _normalize_ts_code_inplace(limit_list_d)
    limit_break_d = _normalize_ts_code_inplace(limit_break_d)
    top_list = _normalize_ts_code_inplace(top_list)

    # 6) 构造 universe（统一底表）
    universe = pd.DataFrame(columns=["ts_code", "name", "close", "pct_chg", "行业", "industry"])

    if isinstance(daily, pd.DataFrame) and (not daily.empty):
        # ts_code
        if "ts_code" in daily.columns:
            universe["ts_code"] = _normalize_ts_code_series(daily["ts_code"])

        # name/close/pct
        name_col = _first_existing_col(daily, ["name", "ts_name", "股票名称", "证券简称"])
        close_col = _first_existing_col(daily, ["close", "收盘价"])
        pct_col = _first_existing_col(daily, ["pct_chg", "pct_change", "change_pct", "涨跌幅"])
        if name_col:
            universe["name"] = _safe_str_series(daily[name_col])
        if close_col:
            universe["close"] = _safe_str_series(daily[close_col])
        if pct_col:
            universe["pct_chg"] = _safe_str_series(daily[pct_col])

    # 保底：确保 ts_code 列存在且归一
    if "ts_code" not in universe.columns:
        universe["ts_code"] = ""
    universe["ts_code"] = _normalize_ts_code_series(universe["ts_code"])

    # 7) name 补齐（从 stock_basic）
    if "name" not in universe.columns:
        universe["name"] = ""
    name_all_blank = True
    if not universe.empty:
        name_all_blank = int((_safe_str_series(universe["name"]) != "").sum()) == 0

    if name_all_blank and (not stock_basic.empty) and (not universe.empty):
        sb_name = _first_existing_col(stock_basic, ["name", "ts_name", "证券简称", "股票名称"])
        if sb_name:
            sb_map = (
                stock_basic[["ts_code", sb_name]]
                .copy()
                .rename(columns={sb_name: "_name"})
            )
            sb_map["ts_code"] = _normalize_ts_code_series(sb_map["ts_code"])
            sb_map["_name"] = _safe_str_series(sb_map["_name"])
            sb_map = sb_map[sb_map["ts_code"] != ""].drop_duplicates(subset=["ts_code"], keep="first")

            universe = universe.merge(sb_map, how="left", on="ts_code")
            universe["name"] = _safe_str_series(universe["name"])
            universe["_name"] = _safe_str_series(universe["_name"])
            mask = universe["name"] == ""
            universe.loc[mask, "name"] = universe.loc[mask, "_name"]
            universe.drop(columns=["_name"], inplace=True, errors="ignore")

    # 8) 行业链路终极合并：优先 stock_basic，其次 daily_basic
    #    用 dict 映射比 merge 更稳（避免重复行/列名冲突）
    def _apply_industry_map(df_src: pd.DataFrame, src_tag: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {"src": src_tag, "used": False, "map_size": 0, "filled": 0}
        nonlocal universe
        if df_src is None or df_src.empty or universe is None or universe.empty:
            return info
        if "ts_code" not in df_src.columns or "industry" not in df_src.columns:
            return info

        src = df_src[["ts_code", "industry"]].copy()
        src["ts_code"] = _normalize_ts_code_series(src["ts_code"])
        src["industry"] = _safe_str_series(src["industry"]).map(_normalize_industry_name)
        src = src[(src["ts_code"] != "") & (src["industry"] != "")]
        src = src.drop_duplicates(subset=["ts_code"], keep="first")

        mp = dict(zip(src["ts_code"], src["industry"]))
        info["map_size"] = int(len(mp))
        if not mp:
            return info

        # universe 保证两列都有
        if "行业" not in universe.columns:
            universe["行业"] = ""
        if "industry" not in universe.columns:
            universe["industry"] = ""

        before = (_safe_str_series(universe["行业"]) != "").sum()
        u_ind = _safe_str_series(universe["行业"])
        # 只填空的
        fill_mask = u_ind == ""
        if fill_mask.any():
            filled_values = universe.loc[fill_mask, "ts_code"].map(mp).fillna("")
            universe.loc[fill_mask, "行业"] = _safe_str_series(filled_values)

        # 同步 industry 字段
        universe["行业"] = _safe_str_series(universe["行业"]).map(_normalize_industry_name)
        universe["industry"] = _safe_str_series(universe["industry"]).map(_normalize_industry_name)
        sync_mask = _safe_str_series(universe["industry"]) == ""
        universe.loc[sync_mask, "industry"] = universe.loc[sync_mask, "行业"]

        after = (_safe_str_series(universe["行业"]) != "").sum()
        info["filled"] = int(after - before)
        info["used"] = True
        return info

    dbg_industry_apply: Dict[str, Any] = {"stock_basic": {}, "daily_basic": {}}
    if not stock_basic.empty:
        dbg_industry_apply["stock_basic"] = _apply_industry_map(stock_basic, "stock_basic")
    if not daily_basic.empty:
        dbg_industry_apply["daily_basic"] = _apply_industry_map(daily_basic, "daily_basic")

    # 9) market（E1/E2/E3）
    e1 = 0
    if isinstance(limit_list_d, pd.DataFrame) and (not limit_list_d.empty):
        if "ts_code" in limit_list_d.columns:
            e1 = int(_normalize_ts_code_series(limit_list_d["ts_code"]).nunique())
        else:
            e1 = int(len(limit_list_d))

    e2 = 0.0
    if isinstance(limit_break_d, pd.DataFrame) and (not limit_break_d.empty):
        if "ts_code" in limit_break_d.columns:
            brk = int(_normalize_ts_code_series(limit_break_d["ts_code"]).nunique())
        else:
            brk = int(len(limit_break_d))
        denom = max(1, (e1 + brk))
        e2 = round(100.0 * brk / denom, 4)

    e3 = 0
    if isinstance(limit_list_d, pd.DataFrame) and (not limit_list_d.empty):
        lb_col = _first_existing_col(limit_list_d, ["lbc", "lb", "consecutive", "连续涨停", "连板数", "连板高度"])
        if lb_col:
            try:
                e3 = int(pd.to_numeric(limit_list_d[lb_col], errors="coerce").fillna(0).max())
            except Exception:
                e3 = 0

    market = {"E1": int(e1), "E2": float(e2), "E3": int(e3)}

    # 10) debug：行业链路关键诊断
    debug_tables = {
        "rows_daily": int(len(daily)) if isinstance(daily, pd.DataFrame) else 0,
        "rows_daily_basic": int(len(daily_basic)) if isinstance(daily_basic, pd.DataFrame) else 0,
        "rows_limit_list_d": int(len(limit_list_d)) if isinstance(limit_list_d, pd.DataFrame) else 0,
        "rows_limit_break_d": int(len(limit_break_d)) if isinstance(limit_break_d, pd.DataFrame) else 0,
        "rows_hot_boards": int(len(hot_boards)) if isinstance(hot_boards, pd.DataFrame) else 0,
        "rows_top_list": int(len(top_list)) if isinstance(top_list, pd.DataFrame) else 0,
        "rows_stock_basic": int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0,
        "rows_universe": int(len(universe)) if isinstance(universe, pd.DataFrame) else 0,
        "universe_cols": list(universe.columns) if isinstance(universe, pd.DataFrame) else [],
        "universe_ts_code_nonempty": int((_safe_str_series(universe.get("ts_code", pd.Series([], dtype=str))) != "").sum())
        if isinstance(universe, pd.DataFrame) and (not universe.empty)
        else 0,
        "universe_industry_nonempty": int((_safe_str_series(universe.get("行业", pd.Series([], dtype=str))) != "").sum())
        if isinstance(universe, pd.DataFrame) and (not universe.empty) and ("行业" in universe.columns)
        else 0,
        "stock_basic_norm": dbg_stock_basic,
        "daily_basic_norm": dbg_daily_basic,
        "industry_apply": dbg_industry_apply,
    }

    # 11) ctx 输出
    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "snapshot_dir": str(snap),
        "snapshot_dir_tried": snap_tried,
        "snapshot_missing": snapshot_missing,
        "debug_step0": debug_tables,

        # 原始表
        "universe": universe,
        "daily": daily,
        "daily_basic": daily_basic,
        "limit_list_d": limit_list_d,
        "limit_break_d": limit_break_d,

        # 兼容不同 step 的读取习惯
        "boards": hot_boards,
        "hot_boards": hot_boards,
        "dragon": top_list,
        "top_list": top_list,

        # 行业主表（已归一）
        "stock_basic": stock_basic,

        "market": market,
    }
    return ctx
