from __future__ import annotations

from typing import Optional

import pandas as pd

from a_top10.config import load_settings
from a_top10.steps.step0_input_layer import step0_build_universe
from a_top10.steps.step1_emotion_gate import step1_emotion_gate
from a_top10.steps.step2_candidate_pool import step2_build_candidates
from a_top10.steps.step6_final_topn import step6_rank_topn
from a_top10.io.writers import write_outputs


def run_pipeline(config_path: str, trade_date: str = "", dry_run: bool = False) -> None:
    """
    最小闭环（V0.2）：
    Step0 数据输入层（读快照/构造ctx）
    Step1 市场情绪过滤器（E1/E2/E3）
    Step2 候选池（涨停池 + H1粗过滤）
    Step6 TopN 输出（占位分数/概率，但产出可复制表格）
    输出：outputs/predict_top10_YYYYMMDD.md + .json + latest.md
    """
    s = load_settings(config_path)
    td = trade_date.strip() or s.trade_date_resolver()

    # Step0
    ctx = step0_build_universe(s, td)

    # Step1
    gate = step1_emotion_gate(s, ctx)

    # Step2 + Step6
    topn: Optional[pd.DataFrame] = None
    if gate.get("pass"):
        candidates = step2_build_candidates(s, ctx)
        topn = step6_rank_topn(s, ctx, candidates)

    # 输出（JSON为准，MD仅展示）
    if not dry_run:
        write_outputs(
            s,
            td,
            ctx=ctx,
            gate=gate,
            topn=topn,
            learn={"updated": False, "note": "v0.2 step0-1-2-6 only"},
        )
