"""
a_top10

A-share Top10 Prediction Engine (Minimal V0)

功能：
- 输入每日快照数据
- 情绪闸门过滤
- 候选池构造
- 输出 Top10 Markdown/JSON

仓库作者：华哥
"""

from __future__ import annotations

# 包版本（后续 Actions / Release 会用到）
__version__ = "0.2.0"

# 对外暴露的主入口
from a_top10.main import run_pipeline

__all__ = [
    "run_pipeline",
    "__version__",
]
