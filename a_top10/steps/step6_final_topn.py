from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from a_top10.config import Settings


def step6_rank_topn(
    s: Settings,
    ctx: Dict[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    冻结口径 Step6（最终TopN）：
    Top10 = sort_by(P * StrengthScore * ThemeBoost)
    V0.2：StrengthScore/ThemeBoost/P 先用占位，但必须输出可用表格。
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=["ts_code", "name", "score", "prob", "board"])

    df = candidates.copy()

    # ---- 占位输出（V0.2）----
    # 后续 Step3/4/5 上线后，把这里替换为真实字段：
    # - StrengthScore（0-100）
    # - ThemeBoost（0-1）
    # - prob（0-1）
    df["StrengthScore"] = 50.0
    df["ThemeBoost"] = 1.0
    df["prob"] = 0.50
    df["score"] = df["prob"] * df["StrengthScore"] * df["ThemeBoost"]

    # 板块字段：V0.2 无法可靠归因，先占位空
    df["board"] = ""

    out = df.sort_values(["score", "prob"], ascending=False).head(int(s.io.topn))
    return out[["ts_code", "name", "score", "prob", "board"]].reset_index(drop=True)
