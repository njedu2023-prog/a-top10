# a_top10/report_writer.py
"""
Top10 Report Writer Module
-------------------------

功能：
- 将 Top10 股票预测结果输出为 Markdown 和 JSON 文件
- 自动生成 latest.md 供 README 展示
- 支持 Gate 未通过时输出空结果提示

输出目录：outputs/
"""

import json
from pathlib import Path
from datetime import datetime

from a_top10.config import prev_a_share_trading_day


OUTPUT_DIR = Path("outputs")


def ensure_output_dir():
    """确保 outputs 目录存在"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_json(top10_list, trade_date):
    """写入 JSON 文件"""
    ensure_output_dir()

    out_file = OUTPUT_DIR / f"predict_top10_{trade_date}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(top10_list, f, ensure_ascii=False, indent=2)

    return out_file


def write_markdown(top10_list, trade_date, gate_passed=True):
    """
    写入 Markdown 文件
    - gate_passed=False 时输出 Gate 未通过提示
    """
    ensure_output_dir()

    out_file = OUTPUT_DIR / f"predict_top10_{trade_date}.md"

    lines = []
    lines.append(f"# Top10 Prediction ({trade_date})\n")

    if not gate_passed or len(top10_list) == 0:
        lines.append("⚠️ Gate 未通过，Top10为空。\n")
    else:
        lines.append("## ✅ 今日预测 Top10 股票\n")
        for i, stock in enumerate(top10_list, 1):
            code = stock.get("ts_code", "UNKNOWN")
            name = stock.get("name", "")
            score = stock.get("score", "")

            lines.append(f"{i}. **{code}** {name}  (score={score})")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_file


def write_latest_md(trade_date, gate_passed=True):
    """
    写入 latest.md（供仓库首页展示）
    """
    ensure_output_dir()

    latest_file = OUTPUT_DIR / "latest.md"

    if not gate_passed:
        content = f"# Top10 Prediction ({trade_date})\n\n⚠️ Gate 未通过，Top10为空。\n"
    else:
        content = (
            f"# Top10 Prediction ({trade_date})\n\n"
            f"📌 查看完整结果：`predict_top10_{trade_date}.md`\n"
        )

    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(content)

    return latest_file


def save_reports(top10_list, trade_date=None, gate_passed=True):
    """
    一键生成所有输出文件：
    - JSON
    - Markdown
    - latest.md
    """
    if trade_date is None:
        trade_date = prev_a_share_trading_day(datetime.now().strftime("%Y%m%d"))

    json_file = write_json(top10_list, trade_date)
    md_file = write_markdown(top10_list, trade_date, gate_passed)
    latest_file = write_latest_md(trade_date, gate_passed)

    print("✅ Outputs generated:")
    print(" -", json_file)
    print(" -", md_file)
    print(" -", latest_file)


if __name__ == "__main__":
    # 测试用例
    demo_top10 = [
        {"ts_code": "000001.SZ", "name": "平安银行", "score": 0.92},
        {"ts_code": "600519.SH", "name": "贵州茅台", "score": 0.88},
    ]

    save_reports(demo_top10, trade_date="20260203", gate_passed=True)
