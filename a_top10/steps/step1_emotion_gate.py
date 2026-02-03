from __future__ import annotations

from typing import Any, Dict

from a_top10.config import Settings


def step1_emotion_gate(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    m = ctx.get("market", {}) or {}
    E1 = int(m.get("E1", 0) or 0)
    E2 = float(m.get("E2", 0.0) or 0.0)
    E3 = int(m.get("E3", 0) or 0)

    # 冻结口径：if (E3 < 3) or (E2 > 35%) or (E1 < 50) => 空仓
    pass_flag = not (
        (E3 < s.emotion_gate.min_max连板高度)
        or (E2 > s.emotion_gate.max_broken_rate)
        or (E1 < s.emotion_gate.min_limit_up_cnt)
    )

    reason = ""
    if not pass_flag:
        reason = f"触发空仓：E3<{s.emotion_gate.min_max连板高度} 或 E2>{s.emotion_gate.max_broken_rate} 或 E1<{s.emotion_gate.min_limit_up_cnt}"

    return {"pass": pass_flag, "E1": E1, "E2": E2, "E3": E3, "reason": reason}
