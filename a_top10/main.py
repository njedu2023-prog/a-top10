from __future__ import annotations

from typing import Dict, Any, Optional

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


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """
    安全读取配置：同时兼容 dict / pydantic / dataclass / 普通对象
    path 示例：'filters.emotion_gate.min_limit_up_cnt'
    """
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur.get(key)
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return default if cur is None else cur


def _safe_get_any(obj: Any, paths: list[str], default: Any = None) -> Any:
    """按多个候选 path 尝试读取，命中一个就返回。"""
    for p in paths:
        v = _safe_get(obj, p, None)
        if v is not None:
            return v
    return default


def _infer_gate_pass(settings: Any, gate: Dict[str, Any], ctx: Any) -> bool:
    """
    兼容旧版 gate 逻辑：
    - 如果 step1 返回了 pass，就直接用
    - 否则尝试读取 configs/default.yml 中 filters.emotion_gate 阈值推导 pass
    - 如果阈值不存在/读取失败，则默认通过（避免“跑绿但空结果”）
    """
    gate = gate or {}
    if "pass" in gate:
        return bool(gate.get("pass"))

    # 取市场指标（优先 gate，其次 ctx.market）
    E1 = int(gate.get("E1", _safe_get(ctx, "market.E1", 0)) or 0)           # 涨停家数
    E2 = float(gate.get("E2", _safe_get(ctx, "market.E2", 0.0)) or 0.0)     # 炸板率（%）
    E3 = int(gate.get("E3", _safe_get(ctx, "market.E3", 0)) or 0)           # 最高连板高度

    # 从配置读取阈值（兼容多种字段命名）
    min_limit_up_cnt = _safe_get_any(
        settings,
        [
            "filters.emotion_gate.min_limit_up_cnt",
            "filters.emotion_gate.min_limit_up_count",
            "filters.emotion_gate.min_zt_cnt",
        ],
        default=None,
    )
    max_broken_rate = _safe_get_any(
        settings,
        [
            "filters.emotion_gate.max_broken_rate",
            "filters.emotion_gate.max_broken_rate_pct",
            "filters.emotion_gate.max_zb_rate",
        ],
        default=None,
    )
    min_max_lianban = _safe_get_any(
        settings,
        [
            "filters.emotion_gate.min_max_lianban",
            "filters.emotion_gate.min_max_lianban_height",
            "filters.emotion_gate.min_lianban_height",
            "filters.emotion_gate.min_max连板高度",  # 兼容你之前可能写进去的旧 key
        ],
        default=None,
    )

    # 配置可能不存在：默认通过
    if min_limit_up_cnt is None and max_broken_rate is None and min_max_lianban is None:
        gate["pass"] = True
        gate["reason"] = (gate.get("reason", "") + " | pass=default(True) (no thresholds)").strip()
        return True

    # 有阈值则按阈值判断（缺的阈值不参与）
    ok = True
    if min_limit_up_cnt is not None:
        try:
            ok = ok and (E1 >= int(min_limit_up_cnt))
        except Exception:
            pass
    if max_broken_rate is not None:
        try:
            ok = ok and (E2 <= float(max_broken_rate))
        except Exception:
            pass
    if min_max_lianban is not None:
        try:
            ok = ok and (E3 >= int(min_max_lianban))
        except Exception:
            pass

    gate["pass"] = bool(ok)

    extra = (
        f" | gate_by_thresholds:"
        f" E1={E1}(min={min_limit_up_cnt}),"
        f" E2={E2}(max={max_broken_rate}),"
        f" E3={E3}(min={min_max_lianban})"
    )
    gate["reason"] = (gate.get("reason", "") + extra).strip()

    return bool(ok)


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
    train_summary: Optional[Dict[str, Any]] = None
    if train_model:
        try:
            train_summary = train_step5_lr(s)
        except Exception as e:
            train_summary = {"ok": False, "error": str(e)}

    # ---- Step0 ----
    ctx = step0_build_universe(s, td)

    # ---- Step1 ----
    gate = step1_emotion_gate(s, ctx)

    # 关键修复：兼容 step1 不返回 pass 的情况，避免 “Gate 未通过 → Top10 为空”
    gate_pass = _infer_gate_pass(s, gate, ctx)

    # ---- Step2–6 ----
    topn_result: Dict[str, pd.DataFrame] = {"topN": None, "full": None}

    if gate_pass:
        candidates = step2_build_candidates(s, ctx)
        strength_df = run_step3(candidates)
        theme_df = run_step4(strength_df)
        prob_df = run_step5(theme_df, s=s)

        # Step6 返回 dict = {"topN": df, "full": df_full}
        topn_result = run_step6_final_topn(prob_df, s=s)

    # ---- 写出结果（新版 writers.py 自动识别 dict 结构）----
    if not dry_run:
        write_outputs(
            settings=s,
            trade_date=td,
            ctx=ctx,
            gate=gate,
            topn=topn_result,  # ✔ 直接传 dict，writers 自动解析
            learn=train_summary or {"updated": False},
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    trade_date = sys.argv[2] if len(sys.argv) > 2 else ""

    # 命令行第三个参数支持 "train"
    train_flag = (len(sys.argv) > 3 and sys.argv[3].lower() == "train")

    run_pipeline(config_path, trade_date, train_model=train_flag)
