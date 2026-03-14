from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from a_top10.config import load_settings
from a_top10.io.writers import write_outputs
from a_top10.steps.step0_input_layer import step0_build_universe
from a_top10.steps.step1_emotion_gate import step1_emotion_gate
from a_top10.steps.step2_candidate_pool import step2_build_candidates
from a_top10.steps.step3_strength_score import run_step3
from a_top10.steps.step4_theme_boost import run_step4
from a_top10.steps.step5_ml_probability import run_step5, train_step5_lr
from a_top10.steps.step6_final_topn import run_step6_final_topn


@contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _force_outputs_dir(settings: Any) -> Path:
    """
    V2 统一强制输出到 a-top10/outputs
    这里只保留最小桥接，不再做旧结构兼容推理。
    """
    out_dir = _repo_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["A_TOP10_OUTPUT_DIR"] = str(out_dir)
    os.environ["TOP10_OUTPUT_DIR"] = str(out_dir)

    io_cfg = getattr(settings, "io", None)
    if io_cfg is not None and hasattr(io_cfg, "outputs_dir"):
        setattr(io_cfg, "outputs_dir", str(out_dir))

    if hasattr(settings, "outputs_dir"):
        setattr(settings, "outputs_dir", str(out_dir))

    return out_dir


def _require_ctx_dict(ctx: Any) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        raise RuntimeError("V2 contract violated: step0_build_universe must return dict ctx")
    return ctx


def _require_gate_dict(gate: Any) -> Dict[str, Any]:
    if not isinstance(gate, dict):
        raise RuntimeError("V2 contract violated: step1_emotion_gate must return dict")
    if "pass" not in gate:
        raise RuntimeError("V2 contract violated: step1_emotion_gate must return {'pass': bool, ...}")
    gate["pass"] = bool(gate["pass"])
    return gate


def _require_dataframe(df: Any, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"V2 contract violated: {name} must return pandas.DataFrame")
    return df


def _normalize_step6_result(obj: Any) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Step6 V2 唯一输出契约：
    {
        "topN": DataFrame,
        "full": DataFrame,
        "limit_up_table": DataFrame | None,
        ...
    }
    """
    if not isinstance(obj, dict):
        raise RuntimeError("V2 contract violated: step6_final_topn must return dict")

    topn_df = obj.get("topN")
    full_df = obj.get("full")
    limit_up_table_df = obj.get("limit_up_table")

    if not isinstance(topn_df, pd.DataFrame):
        raise RuntimeError("V2 contract violated: step6 result missing DataFrame key 'topN'")
    if not isinstance(full_df, pd.DataFrame):
        raise RuntimeError("V2 contract violated: step6 result missing DataFrame key 'full'")
    if limit_up_table_df is not None and not isinstance(limit_up_table_df, pd.DataFrame):
        raise RuntimeError("V2 contract violated: step6 key 'limit_up_table' must be DataFrame or None")

    return {
        "topN": topn_df,
        "topn": topn_df,  # 仅为 writer 外壳输入保留，不是内部双语义
        "full": full_df,
        "limit_up_table": limit_up_table_df,
    }


def run_pipeline(
    config_path: str,
    trade_date: str = "",
    dry_run: bool = False,
    train_model: bool = False,
) -> None:
    """
    V2 主链：
    Step0 -> Step1 -> Step2 -> Step3 -> Step4 -> Step5 -> Step6 -> writers
    """

    settings = load_settings(config_path)
    td = trade_date.strip() or settings.trade_date_resolver()

    out_dir = _force_outputs_dir(settings)

    train_summary: Optional[Dict[str, Any]] = None
    if train_model:
        train_summary = train_step5_lr(settings)

    # Step0
    ctx = _require_ctx_dict(step0_build_universe(settings, td))
    ctx["trade_date"] = td

    # Step1
    gate = _require_gate_dict(step1_emotion_gate(settings, ctx))

    # 默认空输出容器
    topn_result: Dict[str, Optional[pd.DataFrame]] = {
        "topN": None,
        "topn": None,
        "full": None,
        "limit_up_table": None,
    }

    if gate["pass"]:
        # Step2
        candidates = _require_dataframe(step2_build_candidates(settings, ctx), "step2_build_candidates")
        ctx["candidates"] = candidates

        # Step3
        strength_df = _require_dataframe(run_step3(candidates), "run_step3")
        ctx["strength_df"] = strength_df

        # Step4
        ctx = _require_ctx_dict(run_step4(settings, ctx))
        ctx["trade_date"] = td

        theme_df = ctx.get("theme_df")
        if not isinstance(theme_df, pd.DataFrame):
            raise RuntimeError("V2 contract violated: run_step4 must write DataFrame ctx['theme_df']")

        # Step5
        prob_df = _require_dataframe(run_step5(theme_df, s=settings), "run_step5")
        ctx["prob_df"] = prob_df
        ctx["step5_df"] = prob_df

        # Step6
        raw_step6 = run_step6_final_topn(prob_df, s=settings)
        topn_result = _normalize_step6_result(raw_step6)

        ctx["topN_df"] = topn_result["topN"]
        ctx["topn_df"] = topn_result["topN"]
        ctx["final_full_df"] = topn_result["full"]

        ctx.setdefault("debug", {})
        if isinstance(ctx["debug"], dict):
            ctx["debug"]["step5_v2"] = {
                "rows": int(len(prob_df)),
                "columns": list(prob_df.columns),
            }
            ctx["debug"]["step6_v2"] = {
                "topN_rows": int(len(topn_result["topN"])) if topn_result["topN"] is not None else 0,
                "full_rows": int(len(topn_result["full"])) if topn_result["full"] is not None else 0,
            }
    else:
        gate["reason"] = (str(gate.get("reason") or "") + " | pipeline_skipped(step2-6)").strip()

    dbg = ctx.get("debug", {}) if isinstance(ctx, dict) else {}
    print("DEBUG.step5_v2 =", dbg.get("step5_v2") if isinstance(dbg, dict) else None)
    print("DEBUG.step6_v2 =", dbg.get("step6_v2") if isinstance(dbg, dict) else None)

    if not dry_run:
        with _chdir(out_dir.parent):
            write_outputs(
                settings=settings,
                trade_date=td,
                ctx=ctx,
                gate=gate,
                topn=topn_result,
                learn=train_summary or {"updated": False},
            )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    trade_date = sys.argv[2] if len(sys.argv) > 2 else ""
    train_flag = len(sys.argv) > 3 and sys.argv[3].lower() == "train"

    run_pipeline(
        config_path=config_path,
        trade_date=trade_date,
        train_model=train_flag,
    )
