from __future__ import annotations

from typing import Any, Dict, Optional

from a_top10.config import Settings


def _clip01(x: float) -> float:
    """限制 0~1 区间"""
    return max(0.0, min(1.0, x))


def _ewma(prev: Optional[float], now: float, alpha: float = 0.30) -> float:
    """指数移动平均（用于情绪平滑）"""
    if prev is None:
        return now
    return (1 - alpha) * prev + alpha * now


def _regime_state(emotion_smooth: float) -> str:
    """
    仅用于标签解释（不做筛选）。
    可按回测再微调阈值。
    """
    if emotion_smooth >= 0.70:
        return "risk_on"
    if emotion_smooth <= 0.35:
        return "risk_off"
    return "neutral"


def step1_emotion_gate(s: Settings, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step1: Global Regime Factor Service (Emotion Engine v3.1)

    定位（已锁定）：
    - 不对个股定性，不筛选股票（pass 永远 True）
    - 仅生成“全局市场状态/情绪”因子，供后续模块使用：
      * 主供 Step6：融合校准（FinalScore/Prob 映射的尺度因子）
      * 预留 Step4：作为模型特征（全局特征追加）
      * 必要时 Step3：作为尺度因子（整体强度缩放）

    输入（来自 ctx["market"]）：
      - E1: 涨停家数
      - E2: 炸板率（%）
      - E3: 最高连板高度

    输出：
      - regime: 统一的全局因子包（推荐后续模块只消费这个）
      - 同时保留旧字段 EmotionScore/EmotionSmooth/EmotionWeight 以兼容现有主链
    """
    # ---- 读取市场指标 ----
    m = ctx.get("market", {}) or {}
    E1 = int(m.get("E1", 0) or 0)        # 涨停家数
    E2 = float(m.get("E2", 0.0) or 0.0)  # 炸板率（%）
    E3 = int(m.get("E3", 0) or 0)        # 最高连板高度

    # ---- 将市场指标归一化到 0~1 ----
    score_E3 = _clip01(E3 / 5.0)          # 连板高度 >=5 视为满分
    score_E2 = _clip01(1 - E2 / 40.0)     # 炸板率 <=40% 越低越好
    score_E1 = _clip01(E1 / 80.0)         # 涨停家数 >=80 视为满分

    # ---- 合成当日原始情绪（Regime Score）----
    emotion_score = _clip01(
        0.45 * score_E3 +
        0.35 * score_E2 +
        0.20 * score_E1
    )

    # ---- 读取昨日平滑值（优先新结构，其次兼容旧字段）----
    prev_smooth: Optional[float] = None
    try:
        prev_smooth = ctx.get("regime", {}).get("prev_smooth", None)
        if prev_smooth is None and "prev_emotion_smooth" in ctx:
            prev_smooth = float(ctx.get("prev_emotion_smooth"))
        if prev_smooth is not None:
            prev_smooth = float(prev_smooth)
    except Exception:
        prev_smooth = None

    # ---- 情绪平滑（避免跳变）----
    emotion_smooth = _ewma(prev_smooth, emotion_score, alpha=0.30)

    # ---- 权重映射（0.6 ~ 1.2）----
    emotion_weight = round(0.6 + 0.6 * emotion_smooth, 4)

    # ---- 解释文本（用于可追溯/回测归因）----
    reason = (
        f"连板得分:{score_E3:.2f}, "
        f"炸板得分:{score_E2:.2f}, "
        f"活跃度得分:{score_E1:.2f}, "
        f"情绪:{emotion_smooth:.2f}"
    )

    state = _regime_state(emotion_smooth)

    # ---- 统一 Regime 因子包（建议后续模块只消费 regime）----
    regime = {
        "score": round(emotion_score, 4),
        "smooth": round(emotion_smooth, 4),
        "weight": emotion_weight,
        "state": state,  # 仅标签，不用于筛选
        "inputs": {"E1": E1, "E2": E2, "E3": E3},
        "components": {"E3": round(score_E3, 4), "E2": round(score_E2, 4), "E1": round(score_E1, 4)},
        "reason": reason,
        "prev_smooth": round(emotion_smooth, 4),  # 供下一次平滑使用（由主流程持久化）
        # 预留：给 Step4 的全局特征包（字段名稳定）
        "features": {
            "regime_score": round(emotion_score, 4),
            "regime_smooth": round(emotion_smooth, 4),
            "regime_weight": emotion_weight,
        },
        # 预留：给 Step3 的尺度因子（如果启用）
        "scale": {"strength_scale": emotion_weight},
        # 主供：给 Step6 的融合校准因子（如果启用）
        "calib": {"final_score_scale": emotion_weight, "prob_scale": emotion_weight},
    }

    # ---- 返回（兼容旧字段 + 新 regime 结构）----
    return {
        "pass": True,  # v3.x：不空仓、不筛选，仅生成全局因子
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "EmotionScore": regime["score"],
        "EmotionSmooth": regime["smooth"],
        "EmotionWeight": regime["weight"],
        "reason": reason,
        "prev_emotion_smooth": regime["prev_smooth"],  # 兼容旧链路
        "regime": regime,  # ✅ 新契约：推荐后续模块只消费这个
    }
