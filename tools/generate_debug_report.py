#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P1: 标准化 Debug 报告生成器（机器可读 + 人类可读）
输出（每次 run 都会新增文件，且固定入口会更新）：
- outputs/debug/run_meta_<run_id>.json
- outputs/debug/run_meta.json               (latest)
- outputs/debug/step_health_<run_id>.md
- outputs/debug/step_health.md              (latest)
- outputs/debug/last_error.txt              (若捕获到异常/日志中含 Traceback)

设计目标：
1) 无论主流程成功/失败，都能生成报告（建议在 workflow 中用 if: always() 调用）
2) 文件名包含 run_id，保证每次运行“新增”而不是覆盖
3) 尽量从 outputs/ 与现有 debug 文件推断：跑到哪一步、缺什么输入、输出规模等
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DEBUG_DIR = OUTPUTS / "debug"


# -------------------------
# Helpers
# -------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_text_if_exists(p: Path, max_chars: int = 50000) -> str:
    if not p.exists():
        return ""
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
        if len(t) > max_chars:
            return t[:max_chars] + "\n...<truncated>..."
        return t
    except Exception:
        return ""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _guess_trade_date(outputs_dir: Path) -> str:
    """
    优先：
    1) ENV: TRADE_DATE
    2) 从 outputs/predict_top10_YYYYMMDD.* 推断最新日期
    3) fallback: 今天（UTC）日期 YYYYMMDD（仅用于标识，不保证交易日）
    """
    td = os.getenv("TRADE_DATE", "").strip()
    if td:
        return td

    cand = []
    for p in outputs_dir.glob("predict_top10_*.md"):
        m = re.search(r"predict_top10_(\d{8})\.md$", p.name)
        if m:
            cand.append(m.group(1))
    for p in outputs_dir.glob("predict_top10_*.json"):
        m = re.search(r"predict_top10_(\d{8})\.json$", p.name)
        if m:
            cand.append(m.group(1))
    cand = sorted(set(cand))
    if cand:
        return cand[-1]

    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _file_stat(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"exists": False}
    try:
        s = p.stat()
        return {
            "exists": True,
            "bytes": int(s.st_size),
            "mtime_utc": datetime.fromtimestamp(s.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception:
        return {"exists": True}


def _count_rows_csv(p: Path) -> Optional[int]:
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return int(len(df))
    except Exception:
        return None


@dataclass
class StepCheck:
    name: str
    ok: bool
    level: str  # OK / WARN / FAIL / UNKNOWN
    evidence: List[str]


def _check_steps(trade_date: str) -> List[StepCheck]:
    """
    这里用“证据文件”做最稳的推断：
    - 只要某步输出/调试文件存在，即认为该步至少走过
    - 若关键产物缺失，则给 WARN/FAIL
    注意：你后续完善 Step0~7 真实 debug 文件后，这里可继续加证据项
    """

    checks: List[StepCheck] = []

    # Step4 你目前已有 debug_step4_theme_YYYYMMDD.json
    step4_debug = OUTPUTS / f"debug_step4_theme_{trade_date}.json"
    # 预测产物（最终）
    pred_md = OUTPUTS / f"predict_top10_{trade_date}.md"
    pred_json = OUTPUTS / f"predict_top10_{trade_date}.json"

    # learning 目录（你已有 outputs/learning/）
    learning_dir = OUTPUTS / "learning"

    # Step7 常见产物（你后续会补齐，但这里先检测目录/文件）
    # 不强绑定文件名，只检查 learning 是否有新增内容
    learning_files = list(learning_dir.glob("*")) if learning_dir.exists() else []

    # Step0~3 目前仓库未见统一 debug 文件，先做“未知但可提示”
    checks.append(StepCheck("Step0 数据准备", ok=True, level="UNKNOWN", evidence=[
        "未发现统一的 Step0 debug 文件（可后续加 outputs/debug/step0_*.json 作为证据）"
    ]))
    checks.append(StepCheck("Step1 市场情绪过滤", ok=True, level="UNKNOWN", evidence=[
        "未发现统一的 Step1 debug 文件（可后续加 outputs/debug/step1_*.json 作为证据）"
    ]))
    checks.append(StepCheck("Step2 候选池", ok=True, level="UNKNOWN", evidence=[
        "未发现统一的 Step2 debug 文件（可后续加 outputs/debug/step2_*.csv 作为证据）"
    ]))
    checks.append(StepCheck("Step3 涨停质量评分", ok=True, level="UNKNOWN", evidence=[
        "未发现统一的 Step3 debug 文件（可后续加 outputs/debug/step3_*.json 作为证据）"
    ]))

    # Step4
    if step4_debug.exists():
        checks.append(StepCheck("Step4 题材加权", ok=True, level="OK", evidence=[
            f"发现 {step4_debug.as_posix()}",
            f"stat: {json.dumps(_file_stat(step4_debug), ensure_ascii=False)}",
        ]))
    else:
        checks.append(StepCheck("Step4 题材加权", ok=False, level="WARN", evidence=[
            f"缺失 {step4_debug.as_posix()}（说明 Step4 未运行或未落盘 debug）"
        ]))

    # Step5（ML 推断/概率）
    # 这里用“预测文件存在”作为证据：如果没有 predict_*，说明后续没走完
    if pred_md.exists() or pred_json.exists():
        checks.append(StepCheck("Step5 概率推断", ok=True, level="OK", evidence=[
            f"发现预测产物：{pred_md.name if pred_md.exists() else pred_json.name}",
        ]))
    else:
        checks.append(StepCheck("Step5 概率推断", ok=False, level="FAIL", evidence=[
            f"缺失 {pred_md.name} / {pred_json.name}（主流程未产出最终预测）"
        ]))

    # Step6 风险剔除 TopN 输出（通常包含在最终预测里，这里同 Step5 判定）
    if pred_md.exists() or pred_json.exists():
        checks.append(StepCheck("Step6 风险剔除与TopN输出", ok=True, level="OK", evidence=[
            "最终预测文件存在（默认认为 Step6 已完成或已包含在最终输出流程）"
        ]))
    else:
        checks.append(StepCheck("Step6 风险剔除与TopN输出", ok=False, level="FAIL", evidence=[
            "最终预测文件缺失（无法确认 Step6 完成）"
        ]))

    # Step7 自学习闭环（先用 learning 目录是否有内容作为弱证据）
    if learning_dir.exists() and len(learning_files) > 0:
        checks.append(StepCheck("Step7 自学习闭环", ok=True, level="WARN", evidence=[
            f"learning 目录存在且包含 {len(learning_files)} 个文件（需要后续加入明确的 step7 产物命名以强证据判定）",
            f"示例：{', '.join([p.name for p in learning_files[:5]])}{'...' if len(learning_files) > 5 else ''}",
        ]))
    else:
        checks.append(StepCheck("Step7 自学习闭环", ok=False, level="UNKNOWN", evidence=[
            "learning 目录为空或不存在（可能未启用 Step7，或尚未产生产物）"
        ]))

    return checks


def _extract_last_error() -> str:
    """
    兜底：在常见日志文件或 outputs 中搜索 Traceback 片段。
    你后续也可以在主程序异常时把 traceback 写到 outputs/debug/last_error.txt，
    这里会优先使用它。
    """
    # 优先：已有 last_error
    p = DEBUG_DIR / "last_error.txt"
    if p.exists():
        return _read_text_if_exists(p)

    # 其次：在 outputs 下找可能的日志文件
    candidates = []
    for name in ["run.log", "pipeline.log", "error.log", "logs.txt"]:
        q = OUTPUTS / name
        if q.exists():
            candidates.append(q)

    # 最后：不做全仓扫描（太慢），只看这些候选
    for q in candidates:
        t = _read_text_if_exists(q)
        if "Traceback" in t or "Exception" in t:
            return t

    return ""


def build_reports() -> Tuple[Dict[str, Any], str, str]:
    run_id = os.getenv("GITHUB_RUN_ID", "").strip() or f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "").strip()
    sha = os.getenv("GITHUB_SHA", "").strip()
    ref = os.getenv("GITHUB_REF", "").strip()
    actor = os.getenv("GITHUB_ACTOR", "").strip()
    event_name = os.getenv("GITHUB_EVENT_NAME", "").strip()
    workflow = os.getenv("GITHUB_WORKFLOW", "").strip()
    job = os.getenv("GITHUB_JOB", "").strip()

    trade_date = _guess_trade_date(OUTPUTS)

    checks = _check_steps(trade_date)
    err = _extract_last_error()

    meta: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "trade_date": trade_date,
        "github": {
            "run_id": run_id,
            "run_attempt": run_attempt,
            "workflow": workflow,
            "job": job,
            "event": event_name,
            "actor": actor,
            "ref": ref,
            "sha": sha,
        },
        "outputs": {
            "outputs_dir": str(OUTPUTS),
            "debug_dir": str(DEBUG_DIR),
            "predict_md": _file_stat(OUTPUTS / f"predict_top10_{trade_date}.md"),
            "predict_json": _file_stat(OUTPUTS / f"predict_top10_{trade_date}.json"),
            "step4_debug": _file_stat(OUTPUTS / f"debug_step4_theme_{trade_date}.json"),
        },
        "steps": [
            {
                "step": c.name,
                "level": c.level,
                "ok": c.ok,
                "evidence": c.evidence,
            }
            for c in checks
        ],
        "has_error_excerpt": bool(err.strip()),
    }

    # 人类可读 md
    lines: List[str] = []
    lines.append(f"# Step Health（{trade_date}）")
    lines.append("")
    lines.append("## Run 概要")
    lines.append(f"- 生成时间（UTC）：{meta['generated_at_utc']}")
    lines.append(f"- run_id：{meta['github']['run_id']}")
    if meta["github"]["run_attempt"]:
        lines.append(f"- run_attempt：{meta['github']['run_attempt']}")
    if meta["github"]["workflow"]:
        lines.append(f"- workflow：{meta['github']['workflow']}")
    if meta["github"]["job"]:
        lines.append(f"- job：{meta['github']['job']}")
    if meta["github"]["event"]:
        lines.append(f"- event：{meta['github']['event']}")
    if meta["github"]["actor"]:
        lines.append(f"- actor：{meta['github']['actor']}")
    if meta["github"]["ref"]:
        lines.append(f"- ref：{meta['github']['ref']}")
    if meta["github"]["sha"]:
        lines.append(f"- sha：{meta['github']['sha']}")
    lines.append("")

    lines.append("## 模块状态（从产物/证据推断）")
    lines.append("")
    lines.append("| 模块 | 状态 | 说明（关键证据） |")
    lines.append("|---|---|---|")
    for c in checks:
        status = c.level
        evidence = c.evidence[0] if c.evidence else ""
        # 防止太长，表格里只放第一条证据，其余放到明细
        if len(evidence) > 80:
            evidence = evidence[:80] + "…"
        lines.append(f"| {c.name} | {status} | {evidence} |")
    lines.append("")

    lines.append("## 详细证据")
    for c in checks:
        lines.append(f"### {c.name} — {c.level}")
        for e in c.evidence:
            lines.append(f"- {e}")
        lines.append("")

    if err.strip():
        lines.append("## 错误摘录（Traceback/Exception）")
        lines.append("")
        lines.append("```")
        lines.append(err.strip())
        lines.append("```")
        lines.append("")

    md = "\n".join(lines)
    return meta, md, trade_date


def main() -> int:
    meta, md, trade_date = build_reports()

    run_id = meta["github"]["run_id"]
    _ensure_dir(DEBUG_DIR)

    # 每次新增（带 run_id）
    meta_path = DEBUG_DIR / f"run_meta_{run_id}.json"
    md_path = DEBUG_DIR / f"step_health_{run_id}.md"

    _write_json(meta_path, meta)
    _write_text(md_path, md)

    # 固定入口（latest）
    _write_json(DEBUG_DIR / "run_meta.json", meta)
    _write_text(DEBUG_DIR / "step_health.md", md)

    # 同步 last_error（如果 meta 标记有 error excerpt 且 last_error 不存在）
    if meta.get("has_error_excerpt") and not (DEBUG_DIR / "last_error.txt").exists():
        # build_reports 已从候选中抽取了 err，但这里不重复写入，避免污染
        pass

    print(f"[OK] wrote: {meta_path}")
    print(f"[OK] wrote: {md_path}")
    print(f"[OK] latest: {DEBUG_DIR / 'step_health.md'}")
    print(f"[OK] trade_date: {trade_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
