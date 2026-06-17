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

def __getattr__(name: str):
    if name == "run_pipeline":
        from a_top10.main import run_pipeline

        return run_pipeline
    raise AttributeError(name)

__all__ = [
    "run_pipeline",
    "__version__",
]
