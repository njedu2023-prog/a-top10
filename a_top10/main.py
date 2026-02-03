from __future__ import annotations

from typing import Optional
import pandas as pd

from a_top10.config import load_settings

# ---- Pipeline Steps ----
from a_top10.steps.step0_input_layer import step0_build_universe
from a_top10.steps.step1_emotion_gate import step1_emotion_gate
from a_top10.steps.step2_candidate_pool import step2_build_candidates
from a_top10.steps.step3_strength_score import run_step3
from a_top10.steps.step4_theme_boost import run_step4
from a_top10.steps.step5_ml_probability import run_step5
from a_top10.steps.step6_final_topn import run_step6_final_topn

# ---- Output Writer ----
from a_top10.io.writers import write_outputs


def run_pipeline(config_path: str, trade_date: str = "", dry_run: bool = False) -> None:
    """
    完整闭环 Pipeline（Step0 → Step6）

    Step0 数据输入层（读快照/构造 ctx）
    Step1 市场情绪过滤器（E1/E2/E3）
    Step2 候选池（涨停池 + H1 粗过滤）
    Step3 涨停质量评分 StrengthScore（A+B+C）
    Step4 板块/题材 ThemeBoost
    Step5 ML 概率推断 Probability（可训练 LR）
    Step6 TopN 综合排序输出

    输出：
        outputs/predict_top10_YYYYMMDD.md
        outputs/predict_top10_YYYYMMDD.json
        outputs/latest.md
    """
    # ---- Load Settings ----
    s = load_settings(config_path)
    td = trade_date.strip() or s.trade_date_resolver()

    # ---- Step0 ----
    ctx = step0_build_universe(s, td)

    # ---- Step1 ----
    gate = step1_emotion_gate(s, ctx)

    # ---- Step2-6 ----
    topn: Optional[pd.DataFrame] = None

    if gate.get("pass"):
        # Step2 候选池
        candidates = step2_build_candidates(s, ctx)

        # Step3 强度评分
        strength_df = run_step3(candidates)

        # Step4 题材加权
        theme_df = run_step4(strength_df)

        # Step5 ML 推断
        prob_df = run_step5(theme_df, s=s)

        # Step6 最终 TopN
        topn = run_step6_final_topn(prob_df, s=s)

    # ---- 写出结果 ----
    if not dry_run:
        write_outputs(
            s,
            td,
            ctx=ctx,
            gate=gate,
            topn=topn,
            learn={"updated": False, "note": "A-version: full pipeline 0–6"},
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    trade_date = sys.argv[2] if len(sys.argv) > 2 else ""

    run_pipeline(config_path, trade_date)
