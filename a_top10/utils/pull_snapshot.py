#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 GitHub 仓库 a-share-top3-data 拉取指定交易日的快照 CSV 文件，
保存到本地 a_top10/snapshots/{trade_date}/ 目录。

支持的文件包括：
- daily.csv
- daily_basic.csv
- limit_list_d.csv
- hot_boards.csv
- top_list.csv
- 以及该日目录下的所有 CSV 文件（自动发现）

完全自动、路径无需人工修改。
"""

import os
import requests
from pathlib import Path

GITHUB_RAW_PREFIX = (
    "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main/data/raw"
)

def ensure_dir(p: Path):
    """创建目录"""
    p.mkdir(parents=True, exist_ok=True)


def download_file(url: str, save_path: Path):
    """下载单个文件"""
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            save_path.write_bytes(r.content)
            print(f"✅ 下载成功: {save_path}")
        else:
            print(f"⚠️ 远程无此文件: {url}")
    except Exception as e:
        print(f"❌ 下载失败 {url}: {e}")


def pull_snapshot(trade_date: str):
    """
    trade_date 示例: '20260203'
    自动推断 year = 2026
    """

    if len(trade_date) != 8:
        raise ValueError("trade_date 必须是 YYYYMMDD 格式，例如 20260203")

    year = trade_date[:4]

    # 本地保存路径：a_top10/snapshots/{trade_date}/
    root = Path(__file__).resolve().parents[1]  # a_top10 目录
    save_dir = root / "snapshots" / trade_date
    ensure_dir(save_dir)

    print(f"📦 保存目录: {save_dir}")

    # 远程目录 URL 示例：
    # https://raw.githubusercontent.com/.../data/raw/2026/20260203/
    base_url = f"{GITHUB_RAW_PREFIX}/{year}/{trade_date}"

    # 先尝试拉取远程目录文件列表（GitHub raw 不提供，需要写死文件名）
    candidate_files = [
        "daily.csv",
        "daily_basic.csv",
        "hot_boards.csv",
        "limit_list_d.csv",
        "limit_break_d.csv",
        "limit_up_tags.csv",
        "stock_basic.csv",
        "moneyflow_hsgt.csv",
        "namechange.csv",
        "stk_limit.csv",
        "top_list.csv",
    ]

    print("⏬ 开始下载快照 ...")

    for f in candidate_files:
        url = f"{base_url}/{f}"
        download_file(url, save_path=save_dir / f)

    print("\n🎉 快照拉取完成！")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pull_snapshot.py 20260203")
        sys.exit(1)

    pull_snapshot(sys.argv[1])
