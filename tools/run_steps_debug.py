#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/run_steps_debug.py

Top10 系统：模块级单测 / 分段集成 / 全链路回归 的统一调试入口（不修改现有 step 代码）

用法示例：
1) 跑到 step2（分段集成）：
   python tools/run_steps_debug.py --until 2

2) 只跑 step3（要求 step0/1/2 的产物已在 ctx 中；本脚本会尝试先补齐 0..2）：
   python tools/run_steps_debug.py --step 3

3) 指定交易日：
   python tools/run_steps_debug.py --trade-date 20260204 --until 6

4) 自定义 debug 输出目录：
   python tools/run_steps_debug.py --until 6 --outdir outputs/debug

设计目标：
- 先保证“每一步都有输出文件 + 元信息”，再谈模型准不准
- 任何一步输出为空，都要在 debug/meta_stepX.json 里能定位原因（缺字段/行数为0/异常）
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# -----------------------------
# Utils
# -----------------------------
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    return repr(obj)


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(_safe_json(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _df_meta(df: "pd.DataFrame") -> Dict[str, Any]:
    # 行列、缺失率、前几列名
    null_rate = {}
    try:
        for c in df.columns:
            s = df[c]
            # 避免 object 列统计太慢：只算 isna
            nr = float(s.isna().mean())
            if nr > 0:
                null_rate[str(c)] = round(nr, 6)
    except Exception:
        null_rate = {"__error__": "failed_to_compute_null_rate"}

    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns_head": [str(c) for c in list(df.columns)[:30]],
        "null_rate_gt_0": null_rate,
    }


def _dump_artifact(name: str, obj: Any, outdir: Path) -> Dict[str, Any]:
    """
    把 step 输出落盘（支持 DataFrame / dict / list / 基本类型）
    返回 meta 信息
    """
    meta: Dict[str, Any] = {"name": name, "type": type(obj).__name__}

    if pd is not None and isinstance(obj, pd.DataFrame):
        path = outdir / f"{name}.csv"
        _ensure_dir(path.parent)
        obj.to_csv(path, index=False, encoding="utf-8")
        meta.update({"artifact": str(path), "format": "csv"})
        meta.update(_df_meta(obj))
        return meta

    if isinstance(obj, dict) or isinstance(obj, list):
        path = outdir / f"{name}.json"
        _write_json(path, obj)
        meta.update({"artifact": str(path), "format": "json"})
        if isinstance(obj, dict):
            meta["keys_head"] = list(obj.keys())[:50]
        else:
            meta["len"] = len(obj)
        return meta

    # 其他：写成 txt
    path = outdir / f"{name}.txt"
    _write_text(path, repr(obj))
    meta.update({"artifact": str(path), "format": "txt"})
    return meta


def _pick_entrypoint(mod: Any) -> Tuple[Optional[str], Optional[Callable[..., Any]]]:
    """
    自动探测 step 模块的入口函数：
    优先级：run / execute / main / apply / step / transform / process
    """
    candidates = [
        "run",
        "execute",
        "main",
        "apply",
        "step",
        "transform",
        "process",
    ]
    for fn in candidates:
        f = getattr(mod, fn, None)
        if callable(f):
            return fn, f

    # 兜底：找第一个 public callable（排除内置/工具函数）
    for name in dir(mod):
        if name.startswith("_"):
            continue
        f = getattr(mod, name, None)
        if callable(f) and getattr(f, "__module__", "") == getattr(mod, "__name__", ""):
            return name, f

    return None, None


def _call_with_best_effort(fn: Callable[..., Any], ctx: Dict[str, Any], trade_date: str, outdir: Path) -> Any:
    """
    根据 fn 的签名，尽量传入它可能需要的参数：
    - ctx/context/state
    - trade_date/date
    - outdir/debug_dir
    """
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    kwargs: Dict[str, Any] = {}
    if sig is not None:
        for p in sig.parameters.values():
            n = p.name
            if n in ("ctx", "context", "state"):
                kwargs[n] = ctx
            elif n in ("trade_date", "date", "td"):
                kwargs[n] = trade_date
            elif n in ("outdir", "debug_dir", "debug_outdir", "output_dir"):
                kwargs[n] = str(outdir)
            # 允许 step 自己读取环境变量，不强行塞 settings/config（避免签名不匹配）

    # 优先用 kwargs 调用；如果失败再尝试无参/单参
    try:
        return fn(**kwargs) if kwargs else fn()
    except TypeError:
        # 可能只收一个 ctx
        try:
            return fn(ctx)
        except Exception:
            return fn()
    except Exception:
        raise


# -----------------------------
# Runner
# -----------------------------
def run_steps(step: Optional[int], until: Optional[int], trade_date: str, outdir: Path, strict: bool) -> int:
    """
    运行 step0..stepN，且每步落盘 meta + artifact
    """
    _ensure_dir(outdir)

    # 写一份 session meta
    session_meta = {
        "ts": _now(),
        "trade_date": trade_date,
        "mode": {"step": step, "until": until},
        "python": sys.version,
        "cwd": os.getcwd(),
    }
    _write_json(outdir / "meta_session.json", session_meta)

    # step 列表（固定 0..6；step7 输出&闭环一般在主程序/报告层，这里先不强行跑）
    steps = list(range(0, 7))
    if step is not None:
        target_max = step
    elif until is not None:
        target_max = until
    else:
        target_max = 6

    if target_max not in steps:
        raise ValueError(f"target step out of range: {target_max} (valid: 0..6)")

    ctx: Dict[str, Any] = {
        "trade_date": trade_date,
        "debug_outdir": str(outdir),
    }

    for i in range(0, target_max + 1):
        mod_name = f"a_top10.steps.step{i}_"
        # 你仓库里实际文件名是：step0_input_layer.py / step1_emotion_gate.py ...
        # 这里做一个映射，确保 import 成功
        file_map = {
            0: "input_layer",
            1: "emotion_gate",
            2: "candidate_pool",
            3: "strength_score",
            4: "theme_boost",
            5: "ml_probability",
            6: "final_topn",
        }
        mod_name = f"a_top10.steps.step{i}_{file_map[i]}"

        step_meta: Dict[str, Any] = {"step": i, "module": mod_name, "ts": _now()}

        try:
            mod = importlib.import_module(mod_name)
            ep_name, ep = _pick_entrypoint(mod)
            if ep is None:
                raise RuntimeError(
                    f"找不到入口函数（run/execute/main/...）。请在 {mod_name} 里提供 run(ctx, ...) 或 execute(ctx, ...)。"
                )

            step_meta["entrypoint"] = ep_name

            # 执行
            result = _call_with_best_effort(ep, ctx=ctx, trade_date=trade_date, outdir=outdir)

            # 允许两种风格：
            # 1) 返回更新后的 ctx(dict)
            # 2) 返回产物（df/dict/obj），同时 ctx 在内部也被更新
            if isinstance(result, dict) and ("trade_date" in result or "ctx" in result or "data" in result):
                # 认为它是 ctx 或 ctx-like
                ctx.update(result)
                step_meta["result_mode"] = "ctx_update"
                step_meta["ctx_keys_head"] = list(ctx.keys())[:80]
            else:
                step_meta["result_mode"] = "artifact"
                artifact_meta = _dump_artifact(f"step{i}_output", result, outdir)
                step_meta["artifact"] = artifact_meta

            # 把 ctx 快照也写一下（只写键和值的可序列化摘要）
            ctx_snapshot = {k: _safe_json(v) for k, v in ctx.items() if k not in ("__large__",)}
            _write_json(outdir / f"ctx_after_step{i}.json", ctx_snapshot)

            step_meta["status"] = "ok"

        except Exception as e:
            step_meta["status"] = "error"
            step_meta["error"] = repr(e)
            _write_json(outdir / f"meta_step{i}.json", step_meta)

            if strict:
                return 2
            else:
                # 非 strict：继续跑后续（但大概率后面会连锁失败）
                continue

        _write_json(outdir / f"meta_step{i}.json", step_meta)

    # 如果用户只想单跑 stepX：这里仍然会跑 0..X，保证输入不缺
    # 写最终摘要
    summary = {
        "ts": _now(),
        "trade_date": trade_date,
        "finished_until": target_max,
        "outdir": str(outdir),
    }
    _write_json(outdir / "meta_summary.json", summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Top10 steps debug runner")
    parser.add_argument("--trade-date", default=os.getenv("TRADE_DATE", "").strip(), help="YYYYMMDD（默认读取环境变量 TRADE_DATE）")
    parser.add_argument("--step", type=int, default=None, help="只测试到某一步（会自动补齐跑 0..step）")
    parser.add_argument("--until", type=int, default=None, help="分段集成：跑到 until（会跑 0..until）")
    parser.add_argument("--outdir", default="outputs/debug", help="debug 输出目录")
    parser.add_argument("--strict", action="store_true", help="任一步报错立即退出（建议 CI 用）")

    args = parser.parse_args()

    trade_date = (args.trade_date or "").strip()
    if not trade_date:
        # 最保守：用今天日期（注意：你主程序如果有交易日解析器，这里不替代它）
        trade_date = datetime.now().strftime("%Y%m%d")

    outdir = Path(args.outdir)

    if args.step is not None and args.until is not None:
        print("参数冲突：--step 和 --until 只能选一个", file=sys.stderr)
        return 2

    rc = run_steps(step=args.step, until=args.until, trade_date=trade_date, outdir=outdir, strict=args.strict)
    if rc == 0:
        print(f"[OK] debug outputs written to: {outdir}")
    else:
        print(f"[FAIL] see meta_*.json under: {outdir}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
