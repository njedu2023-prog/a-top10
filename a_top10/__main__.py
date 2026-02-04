# a_top10/__main__.py
from __future__ import annotations

import argparse
from pathlib import Path

from a_top10.main import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="a_top10")
    parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        default="configs/default.yml",
        help="配置文件路径（默认：configs/default.yml）",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config 不存在: {config_path.resolve()}")

    # run_pipeline 需要 config_path 参数
    run_pipeline(str(config_path))


if __name__ == "__main__":
    main()
