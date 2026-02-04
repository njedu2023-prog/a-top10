#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 GitHub 数据仓库 (a-share-top3-data) 拉取每日快照 CSV。
不会依赖本地路径，100% 使用 GitHub Raw URL 读取。
"""

import os
import requests
from pathlib import Path
from datetime import datetime

# 你的数据仓库信息
GITHUB_USER = "njedu2023-prog"
DATA_REPO = "a-share-top3-data"
BRANCH = "main"

# 快照目录
LOCAL_SNAPSHOT_ROOT = Path("snapshots")  # 存在 a-top10 仓库内部


def github_raw_url(trade_date: str, file_name: str) -> str:
    """
    自动构造 raw.githubusercontent.com 下载链接
    例如：
    https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main/snapshots/20250203/daily.csv
    """
    return (
        f"https://raw.githubusercontent.com/"
        f"{GITHUB_USER}/{DATA_REPO}/{BRANCH}/snapshots/{trade_date}/{file_name}"
    )


def download_file(url: str, save_path: Path):
    """下载文件并保存"""
    print(f"Downloading: {url}")
    resp = requests.get(url)
    if resp.status_code == 200:
        save_path.write_bytes(resp.content)
        print(f"Saved → {save_path}")
    else:
        print(f"⚠️ 文件不存在或无法访问：{url}")


def pull_snapshot(trade_date: str):
    """拉取某个交易日的全部快照文件"""
    files = [
        "daily.csv",
        "daily_basic.csv",
        "limit_list_d.csv",
        "hot_boards.csv",
        "top_list.csv",
    ]

    target_dir = LOCAL_SNAPSHOT_ROOT / trade_date
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        url = github_raw_url(trade_date, f)
        save_file = target_dir / f
        download_file(url, save_file)

    print(f"📦 完成快照拉取：{trade_date} → {target_dir}")


if __name__ == "__main__":
    # 自动用今天的日期（你也可以指定）
    today = datetime.now().strftime("%Y%m%d")
    pull_snapshot(today)
