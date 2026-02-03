from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from a_top10.config import Settings


def step2_build_candidates(s: Settings, ctx: Dict[str, Any]) -> pd.DataFrame:
    """
    冻结口径 Step2（硬条件筛选）：
    - 今日涨停（收盘封死）
    - 非ST非退市风险（H1） —— V0.2 暂用名称字段粗过滤（含ST/退）做最小闭环
    - 非极端一字板（可选） —— V0.2 不做（缺分钟/盘口数据）
    输出：CandidateSet（30~200只股票，实际取决于当日涨停数量）
    """
    limit_df: pd.DataFrame = ctx.get("limit_list_d", pd.DataFrame()).copy()
    if limit_df is None or limit_df.empty:
        return pd.DataFrame(columns=["ts_code", "name"])

    # 字段兼容：ts_code/name 可能存在也可能不完整
    if "ts_code" not in limit_df.columns:
        # 兜底：找类似列
        for c in limit_df.columns:
            if c.lower() == "ts_code":
                limit_df.rename(columns={c: "ts_code"}, inplace=True)
                break
    if "name" not in limit_df.columns:
        # 有些快照可能叫 '股票名称' 之类，这里先不强求，缺就空
        limit_df["name"] = ""

    cand = limit_df[["ts_code", "name"]].drop_duplicates().reset_index(drop=True)

    # H1 粗过滤：剔除 ST / *ST / 退 / 退市
    def _is_bad_name(x: Any) -> bool:
        t = str(x or "")
        return ("ST" in t) or ("*ST" in t) or ("退" in t) or ("退市" in t)

    cand = cand[~cand["name"].apply(_is_bad_name)].reset_index(drop=True)

    return cand
