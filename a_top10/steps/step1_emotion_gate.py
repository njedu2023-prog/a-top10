from __future__ import annotations
from typing import Any, Dict
from a_top10.config import Settings


def _clip01(x: float) -> float:
    """限制 0~1 区间"""
    return max(0.0, min(1.0, x))


def _ewma(prev: float, now: float, alpha: float = 0.30) -> float:
    """情绪平滑：指数移动平均"""
    if prev is None:
        return now
    return (1 - alpha) * prev + alpha * now


def step1_emotion_gate(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emotion Engine v3.0
    - 不再“空仓”
    - 生成连续情绪分数 EmotionScore（0~1）
    - 平滑后 EmotionSmooth（0~1）
    - 推导情绪权重 EmotionWeight（0.6~1.2）
    """

    # ---- 读取市场指标 ----
    m = ctx.get("market", {}) or {}
    E1 = int(m.get("E1", 0) or 0)     # 涨停家数
    E2 = float(m.get("E2", 0.0) or 0.0)  # 炸板率（%）
    E3 = int(m.get("E3", 0) or 0)     # 最高连板高度

    # ---- 将市场指标归一化到 0~1 ----
    score_E3 = _clip01(E3 / 5.0)          # 连板高度，>=5 视为满分
    score_E2 = _clip01(1 - E2 / 40.0)     # 炸板率<=40% 为满分，越高越差
    score_E1 = _clip01(E1 / 80.0)         # 涨停家数>=80 为满分

    # ---- 权重合成（Regime Score） ----
    # 专业量化体系的权重：E3 连板最重要，E2 承接次之
    emotion_score = (
        0.45 * score_E3 +
        0.35 * score_E2 +
        0.20 * score_E1
    )
    emotion_score = _clip01(emotion_score)

    # ---- 读取昨日平滑值（若无则取 None） ----
    prev_smooth = None
    if "prev_emotion_smooth" in ctx:
        try:
            prev_smooth = float(ctx["prev_emotion_smooth"])
        except Exception:
            prev_smooth = None

    # ---- 情绪平滑（避免情绪跳变） ----
    emotion_smooth = _ewma(prev_smooth, emotion_score, alpha=0.30)

    # ---- 得到最终情绪权重（0.6 ~ 1.2）----
    emotion_weight = 0.6 + 0.6 * emotion_smooth
    emotion_weight = round(emotion_weight, 4)

    # ---- 市场说明 ----
    reason = (
        f"连板得分:{score_E3:.2f}, "
        f"炸板得分:{score_E2:.2f}, "
        f"活跃度得分:{score_E1:.2f}, "
        f"情绪:{emotion_smooth:.2f}"
    )

    # ---- 返回结构兼容旧版（不再空仓）----
    return {
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "EmotionScore": round(emotion_score, 4),
        "EmotionSmooth": round(emotion_smooth, 4),
        "EmotionWeight": emotion_weight,
        "reason": reason,
        # 提供下一次平滑用
        "prev_emotion_smooth": round(emotion_smooth, 4),
    }
