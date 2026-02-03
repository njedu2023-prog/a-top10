#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 top3 数据仓库复制 snapshot 文件夹到当前 top10 工程。
只做“数据同步”，不影响 step0~step6 的任何逻辑。
"""

from pathlib import Path
import shutil


def pull_snapshot(trade_date: str,
                  top3_repo: str = "https://github.com/njedu2023-prog/a-share-top3-data",
                  local_top3: str = None,
                  top10_root: str = ".") -> dict:
    """
    trade_date: '20240203' 这样的日期字符串
    top3_repo: 你 top3 的 GitHub 仓库（仅用于提示，不实际 clone）
    local_top3: 你本地已经 clone 好的 top3 仓库路径，例如：/Users/hua/a-share-top3-data
    top10_root: 当前 top10 工程根路径
    """

    if local_top3 is None:
        return {
            "ok": False,
            "reason": "❌ 缺少 local_top3，本地 top3 仓库路径必须提供。\n"
                      "示例：pull_snapshot('20240203', local_top3='/Users/hua/a-share-top3-data')"
        }

    p3 = Path(local_top3) / "data" / "raw" / trade_date     # top3 snapshot 路径
    p10 = Path(top10_root) / "data" / trade_date            # top10 snapshot 路径

    if not p3.exists():
        return {
            "ok": False,
            "reason": f"❌ 找不到 top3 snapshot：{p3}\n请检查本地是否已 clone 最新数据。"
        }

    p10.mkdir(parents=True, exist_ok=True)

    # ------- 复制所有 csv -------
    copied = []
    for f in p3.glob("*.csv"):
        tgt = p10 / f.name
        shutil.copy2(f, tgt)
        copied.append(f.name)

    return {
        "ok": True,
        "trade_date": trade_date,
        "top3_path": str(p3),
        "top10_path": str(p10),
        "copied": copied,
    }


if __name__ == "__main__":
    print("pull_snapshot.py ready.")
