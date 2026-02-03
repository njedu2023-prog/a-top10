from __future__ import annotations

from a_top10.config import load_settings
from a_top10.steps.step0_input_layer import step0_build_universe
from a_top10.steps.step1_emotion_gate import step1_emotion_gate
from a_top10.io.writers import write_outputs


def run_pipeline(config_path: str, trade_date: str = "", dry_run: bool = False) -> None:
    s = load_settings(config_path)
    td = trade_date.strip() or s.trade_date_resolver()

    ctx = step0_build_universe(s, td)
    gate = step1_emotion_gate(s, ctx)

    # V0.1：先不做 Step2~Step7，TopN 先为空占位
    topn = None

    if not dry_run:
        write_outputs(s, td, ctx=ctx, gate=gate, topn=topn, learn={"updated": False, "note": "v0.1 step0-1 only"})
