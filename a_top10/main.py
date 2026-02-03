from __future__ import annotations

from a_top10.config import load_settings
from a_top10.steps.step0_input_layer import step0_build_universe
from a_top10.steps.step1_emotion_gate import step1_emotion_gate
from a_top10.steps.step2_candidate_pool import step2_build_candidates
from a_top10.steps.step3_strength_score import step3_strength_score
from a_top10.steps.step4_theme_boost import step4_theme_boost
from a_top10.steps.step5_ml_prob import step5_predict_prob
from a_top10.steps.step6_risk_and_topn import step6_risk_and_topn
from a_top10.steps.step7_learn_loop import step7_learn
from a_top10.io.writers import write_outputs

def run_pipeline(config_path: str, trade_date: str = "", dry_run: bool = False) -> None:
    s = load_settings(config_path)
    td = trade_date.strip() or s.trade_date_resolver()

    # Step0: Universe（统一底表）
    ctx = step0_build_universe(s, td)

    # Step1: 情绪开关（可能直接空仓）
    gate = step1_emotion_gate(s, ctx)
    if not gate["pass"]:
        if not dry_run:
            write_outputs(s, td, ctx=ctx, gate=gate, topn=None)
        return

    # Step2: 候选池
    cand = step2_build_candidates(s, ctx)

    # Step3: 强度分
    cand = step3_strength_score(s, ctx, cand)

    # Step4: 题材加权
    cand = step4_theme_boost(s, ctx, cand)

    # Step5: 概率预测
    cand = step5_predict_prob(s, ctx, cand)

    # Step6: 风险剔除 + TopN
    topn = step6_risk_and_topn(s, ctx, cand)

    # Step7: 自学习闭环（先可为空实现）
    learn = step7_learn(s, td, ctx, cand, topn)

    if not dry_run:
        write_outputs(s, td, ctx=ctx, gate=gate, topn=topn, learn=learn)
