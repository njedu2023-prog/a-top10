from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import os
from contextlib import contextmanager

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

    # 1) step1 已明确给 pass
    if "pass" in gate:
        return bool(gate.get("pass"))

    # 2) 取市场指标（优先 gate，其次 ctx.market）
    E1 = int(gate.get("E1", _safe_get(ctx, "market.E1", 0)) or 0)       # 涨停家数
    E2 = float(gate.get("E2", _safe_get(ctx, "market.E2", 0.0)) or 0.0) # 炸板率（%）
    E3 = int(gate.get("E3", _safe_get(ctx, "market.E3", 0)) or 0)       # 最高连板高度

    # 3) 从配置读取阈值（兼容多种字段命名）
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
            "filters.emotion_gate.min_max连板高度",
        ],
        default=None,
    )

    # 4) 配置可能不存在：默认通过
    if min_limit_up_cnt is None and max_broken_rate is None and min_max_lianban is None:
        gate["pass"] = True
        gate["reason"] = (gate.get("reason", "") + " | pass=default(True) (no thresholds)").strip()
        return True

    # 5) 有阈值则按阈值判断（缺的阈值不参与）
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


@contextmanager
def _chdir(path: Path):
    """临时切换工作目录，确保相对路径写入到 repo 根目录。"""
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def _force_outputs_dir(settings: Any) -> Path:
    """
    强制输出目录固定为 /outputs （即 a-top10/outputs/）。
    - 通过 __file__ 推断 repo 根目录
    - 尝试写回 settings 里常见字段名（兼容不同配置结构）
    - 同时设置环境变量，方便 writers 或其他模块读取
    """
    repo_root = Path(__file__).resolve().parents[1]  # .../a-top10
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["A_TOP10_OUTPUT_DIR"] = str(out_dir)
    os.environ["TOP10_OUTPUT_DIR"] = str(out_dir)

    candidates = [
        "output_dir",
        "outputs_dir",
        "out_dir",
        "output_path",
        "outputs_path",
        "output_root",
        "outputs_root",
        "result_dir",
        "results_dir",
    ]
    for key in candidates:
        try:
            if hasattr(settings, key):
                setattr(settings, key, str(out_dir))
        except Exception:
            pass

    nested_paths = [
        "io.output_dir",
        "io.outputs_dir",
        "io.out_dir",
        "paths.output_dir",
        "paths.outputs_dir",
        "paths.out_dir",
        "writer.output_dir",
        "writer.outputs_dir",
        "writer.out_dir",
        "outputs.dir",
        "outputs.path",
    ]
    for p in nested_paths:
        parts = p.split(".")
        cur = settings
        ok = True

        for k in parts[:-1]:
            try:
                if isinstance(cur, dict):
                    cur = cur.get(k)
                else:
                    cur = getattr(cur, k, None)
            except Exception:
                cur = None
            if cur is None:
                ok = False
                break

        if not ok:
            continue

        last = parts[-1]
        try:
            if isinstance(cur, dict):
                cur[last] = str(out_dir)
            else:
                if hasattr(cur, last):
                    setattr(cur, last, str(out_dir))
        except Exception:
            pass

    return out_dir


def _ensure_step4_debug_compat(ctx: Dict[str, Any]) -> None:
    """
    兼容旧字段：很多老代码/日志会打印 ctx.debug.step4
    但新版 Step4 实际写 ctx.debug.step4_theme
    这里做一次桥接：step4 不存在时，用 step4_theme 填一份到 step4
    """
    dbg = ctx.get("debug")
    if not isinstance(dbg, dict):
        return
    if dbg.get("step4") is None and isinstance(dbg.get("step4_theme"), dict):
        dbg["step4"] = dbg["step4_theme"]


def _ensure_step5_v2_fields(prob_df: pd.DataFrame) -> pd.DataFrame:
    """
    入口层兜底：
    即使 Step5 还没完全升级，入口也尽量补齐 V2 字段，避免 Step6 断链。
    规则：
    - prob_final 缺失 -> fallback 到 Probability / prob / probability
    - Probability 缺失 -> 回填为 prob_final
    - prob_rule / prob_ml 缺失 -> 建空列
    """
    if not isinstance(prob_df, pd.DataFrame):
        return pd.DataFrame()

    df = prob_df.copy()

    lower_map = {str(c).lower(): c for c in df.columns}

    def find_col(cands: list[str]) -> Optional[str]:
        for c in cands:
            hit = lower_map.get(c.lower())
            if hit is not None:
                return hit
        return None

    prob_final_col = find_col(["prob_final"])
    if prob_final_col is None:
        fallback = find_col(["Probability", "probability", "prob"])
        if fallback is not None:
            df["prob_final"] = pd.to_numeric(df[fallback], errors="coerce").fillna(0.0)
        else:
            df["prob_final"] = 0.0
    elif prob_final_col != "prob_final":
        df["prob_final"] = pd.to_numeric(df[prob_final_col], errors="coerce").fillna(0.0)
    else:
        df["prob_final"] = pd.to_numeric(df["prob_final"], errors="coerce").fillna(0.0)

    if "Probability" not in df.columns:
        df["Probability"] = df["prob_final"]
    else:
        df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce").fillna(df["prob_final"])

    if "prob" not in df.columns:
        df["prob"] = df["prob_final"]
    else:
        df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(df["prob_final"])

    if "prob_rule" not in df.columns:
        df["prob_rule"] = 0.0
    else:
        df["prob_rule"] = pd.to_numeric(df["prob_rule"], errors="coerce").fillna(0.0)

    if "prob_ml" not in df.columns:
        df["prob_ml"] = pd.Series([None] * len(df), index=df.index, dtype="object")

    if "prob_ml_available" not in df.columns:
        df["prob_ml_available"] = pd.to_numeric(df["prob_ml"], errors="coerce").notna()

    if "prob_src" not in df.columns:
        df["prob_src"] = df["prob_ml_available"].map(lambda x: "ml" if bool(x) else "rule_only")

    if "prob_fusion_mode" not in df.columns:
        df["prob_fusion_mode"] = df["prob_ml_available"].map(lambda x: "ml_first" if bool(x) else "fallback_rule")

    for c in ["prob_rule", "prob_final", "prob", "Probability"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(0.0, 1.0).fillna(0.0)

    return df


def _normalize_topn_result(topn_result: Any) -> Dict[str, Optional[pd.DataFrame]]:
    """
    writers 的输入契约再收紧一层：
    - 始终返回 dict
    - 始终包含 topN / topn / full / limit_up_table
    - 非 DataFrame 内容一律转为空
    """
    base: Dict[str, Optional[pd.DataFrame]] = {
        "topN": None,
        "topn": None,
        "full": None,
        "limit_up_table": None,
    }

    if not isinstance(topn_result, dict):
        return base

    for k in ["topN", "topn", "full", "limit_up_table"]:
        v = topn_result.get(k)
        base[k] = v if isinstance(v, pd.DataFrame) else None

    if base["topN"] is None and base["topn"] is not None:
        base["topN"] = base["topn"]
    if base["topn"] is None and base["topN"] is not None:
        base["topn"] = base["topN"]

    return base


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

    # ---- 强制输出目录为 a-top10/outputs ----
    out_dir = _force_outputs_dir(s)

    # ---- Optional: train LR model ----
    train_summary: Optional[Dict[str, Any]] = None
    if train_model:
        try:
            train_summary = train_step5_lr(s)
        except Exception as e:
            train_summary = {"ok": False, "error": str(e)}

    # ---- Step0 ----
    ctx = step0_build_universe(s, td)
    if not isinstance(ctx, dict):
        ctx = {}
    ctx["trade_date"] = td

    # ---- Step1 ----
    gate = step1_emotion_gate(s, ctx) or {}

    # 关键修复：兼容 step1 不返回 pass 的情况，避免 “Gate 未通过 → Top10 为空”
    gate_pass = _infer_gate_pass(s, gate, ctx)

    # ---- Step2–6 ----
    topn_result: Dict[str, Optional[pd.DataFrame]] = {
        "topN": None,
        "topn": None,
        "full": None,
        "limit_up_table": None,
    }

    if gate_pass:
        # Step2
        candidates = step2_build_candidates(s, ctx)
        ctx["candidates"] = candidates

        # Step3
        strength_df = run_step3(candidates)

        # Step3 结果写入 ctx，供 Step4 主线入口读取
        ctx["strength_df"] = strength_df

        # Step4 按主线接口运行，返回更新后的 ctx
        ctx = run_step4(s, ctx)
        if not isinstance(ctx, dict):
            ctx = {}

        # 强制补回 trade_date，防止中途被覆盖掉
        ctx["trade_date"] = td

        # 兼容：把 step4_theme 映射到 step4（避免 DEBUG.step4=None）
        _ensure_step4_debug_compat(ctx)

        # 优先用 Step4 的输出 theme_df（没有再退回 strength_df）
        theme_df = ctx.get("theme_df")
        if not isinstance(theme_df, pd.DataFrame) or theme_df.empty:
            theme_df = ctx.get("strength_df", strength_df)

        # Step5: 进入 V2 概率语义层
        prob_df = run_step5(theme_df, s=s)
        prob_df = _ensure_step5_v2_fields(prob_df)

        # 关键：把 Step5 V2 结果显式放回 ctx，方便 writer/debug/后续步骤读取
        ctx["prob_df"] = prob_df
        ctx["step5_df"] = prob_df

        # Step6: 纯终排器，只认 prob_final 为主口径
        raw_topn_result = run_step6_final_topn(prob_df, s=s)
        topn_result = _normalize_topn_result(raw_topn_result)

        # 兼容桥接：把 full/topN 再挂回 ctx，避免旧 writer 或旧调试逻辑拿不到
        if isinstance(topn_result.get("full"), pd.DataFrame):
            ctx["final_full_df"] = topn_result["full"]

        if isinstance(topn_result.get("topN"), pd.DataFrame):
            ctx["topn_df"] = topn_result["topN"]
            ctx["topN_df"] = topn_result["topN"]

        # 附带 step6 元信息，方便后续排查
        ctx.setdefault("debug", {})
        if isinstance(ctx["debug"], dict):
            ctx["debug"]["step5_v2"] = {
                "columns": list(prob_df.columns),
                "rows": int(len(prob_df)),
                "has_prob_rule": "prob_rule" in prob_df.columns,
                "has_prob_ml": "prob_ml" in prob_df.columns,
                "has_prob_final": "prob_final" in prob_df.columns,
                "has_prob": "prob" in prob_df.columns,
                "has_Probability": "Probability" in prob_df.columns,
            }
            ctx["debug"]["step6_v2"] = {
                "meta": raw_topn_result.get("meta", {}) if isinstance(raw_topn_result, dict) else {},
                "warnings": raw_topn_result.get("warnings", []) if isinstance(raw_topn_result, dict) else [],
                "topN_rows": int(len(topn_result["topN"])) if isinstance(topn_result.get("topN"), pd.DataFrame) else 0,
                "full_rows": int(len(topn_result["full"])) if isinstance(topn_result.get("full"), pd.DataFrame) else 0,
            }

    else:
        gate["reason"] = (gate.get("reason", "") + " | pipeline_skipped(step2-6)").strip()

    # ---- 写出结果（新版 writers.py 自动识别 dict 结构）----
    # 额外保险：切到 repo_root 再写，避免 writers 使用相对路径写到别处
    dbg = ctx.get("debug", {}) if isinstance(ctx, dict) else {}
    print("DEBUG.step4 =", (dbg.get("step4") if isinstance(dbg, dict) else None))
    print("DEBUG.step4_theme =", (dbg.get("step4_theme") if isinstance(dbg, dict) else None))
    print("DEBUG.step5_v2 =", (dbg.get("step5_v2") if isinstance(dbg, dict) else None))
    print("DEBUG.step6_v2 =", (dbg.get("step6_v2") if isinstance(dbg, dict) else None))

    if not dry_run:
        with _chdir(out_dir.parent):
            write_outputs(
                settings=s,
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
    train_flag = (len(sys.argv) > 3 and sys.argv[3].lower() == "train")

    run_pipeline(config_path, trade_date, train_model=train_flag)
