from __future__ import annotations

from typing import Optional, Dict
import pandas as pd

from a_top10.config import load_settings

# ---- Pipeline Steps ----
from a_top10.steps.step0_input_layer import step0_build_universe
from a_top10.steps.step1_emotion_gate import step1_emotion_gate
from a_top10.steps.step2_candidate_pool import step2_build_candidates
from a_top10.steps.step3_strength_score import run_step3
from a_top10.steps.step4_theme_boost import run_step4
from a_top10.steps.step5_ml_probability import run_step5, train_step5_lr
from a_top10.steps.step6_final_topn import run_step6_final_topn

# ---- Output Writer ----
from a_top10.io.writers import write_outputs


def run_pipeline(
    config_path: str,
    trade_date: str = "",
    dry_run: bool = False,
    train_model: bool = False,
) -> None:
    """
    完整闭环 Pipeline（Step0 → Step6）
    """

    # ---- Load Settings ----
    s = load_settings(config_path)
    td = trade_date.strip() or s.trade_date_resolver()

    # ---- Optional: train LR model ----
    train_summary = None
    if train_model:
        try:
            train_summary = train_step5_lr(s)
        except Exception as e:
            train_summary = {"ok": False, "error": str(e)}

    # ---- Step0 ----
    ctx = step0_build_universe(s, td)

    # ---- Step1 ----
    gate = step1_emotion_gate(s, ctx)

    # ---- Step2-6 ----
    topn_df = None
    full_df = None

    if gate.get("pass"):
        candidates = step2_build_candidates(s, ctx)
        strength_df = run_step3(candidates)
        theme_df = run_step4(strength_df)
        prob_df = run_step5(theme_df, s=s)

        # Step6 返回 dict
        result: Dict[str, pd.DataFrame] = run_step6_final_topn(prob_df, s=s)
        topn_df = result.get("topN")
        full_df = result.get("full")

    # ---- 写出结果 ----
    if not dry_run:
        write_outputs(
            s,
            td,
            ctx=ctx,
            gate=gate,
            topn=topn_df,     # 只写 Top10 给用户
            full=full_df,     # 新增：写 full 排序表
            learn=train_summary or {"updated": False},
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    trade_date = sys.argv[2] if len(sys.argv) > 2 else ""
    
    # 命令行第三个参数支持 "train"
    train_flag = (len(sys.argv) > 3 and sys.argv[3].lower() == "train")

    run_pipeline(config_path, trade_date, train_model=train_flag)
