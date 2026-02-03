from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from a_top10.io.paths import OutputPaths
from a_top10.config import Settings


def _safe_json(obj: Any) -> Any:
    # 把 dataframe/时间等转成可json的形式
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def write_outputs(
    s: Settings,
    trade_date: str,
    ctx: Dict[str, Any],
    gate: Dict[str, Any],
    topn: Optional[pd.DataFrame],
    learn: Optional[Dict[str, Any]] = None,
) -> None:
    out = OutputPaths(root=s.io.outputs_dir)

    # -------- JSON（机器用，绝不反解析MD）--------
    payload = {
        "trade_date": trade_date,
        "gate": gate,
        "market": ctx.get("market", {}),
        "topn": [] if topn is None else topn.to_dict(orient="records"),
        "learn": learn or {},
        "meta": {
            "version": s.version,
        },
    }
    jp = out.json_path(trade_date)
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_safe_json), encoding="utf-8")

    # -------- MD（人看）--------
    lines = []
    lines.append(f"# A股 Top10（{trade_date}）")
    lines.append("")
    lines.append("## Step1 市场情绪过滤器（E类优先）")
    lines.append(f"- 通过：**{gate.get('pass')}**")
    lines.append(f"- E1 涨停家数：{gate.get('E1')}")
    lines.append(f"- E2 炸板率：{gate.get('E2')}")
    lines.append(f"- E3 连板高度：{gate.get('E3')}")
    if not gate.get("pass"):
        lines.append(f"- 原因：{gate.get('reason','')}")
        lines.append("")
        lines.append("> 情绪不满足：直接空仓，不输出 TopN。")
    else:
        lines.append("")
        lines.append("## Step6 最终 TopN 输出（V0.1 最小闭环）")
        if topn is None or topn.empty:
            lines.append("> 暂无 TopN（占位输出）。")
        else:
            # 用 pandas to_markdown（若环境无 tabulate，也能退化）
            try:
                lines.append(topn.to_markdown(index=False))
            except Exception:
                lines.append(topn.to_csv(index=False))

    mp = out.md_path(trade_date)
    mp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    # latest.md 固定入口（直接复制内容，不搞软链）
    out.latest_md_path().write_text(mp.read_text(encoding="utf-8"), encoding="utf-8")
