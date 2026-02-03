"""
a_top10.io

负责所有输入输出模块：

- writers.py: 输出 markdown/json
- readers.py: 未来扩展读取接口
"""

from __future__ import annotations

# 暴露主要写出函数（方便外部调用）
from a_top10.io.writers import write_outputs

__all__ = [
    "write_outputs",
]
