#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/debug_print_snapshot.py

✅ 旁路调试工具（稳定版）：
- 不修改任何 step 文件
- 单独运行即可打印 pipeline 各阶段输出
- 兼容文件缺失 / 空文件 / 编码异常 / 分隔符异常（csv/tsv）

用法：
  python tools/debug_print_snapshot.py
可选：
  python tools/debug_print_snapshot.py --n 20
  python tools/debug_print_snapshot.py --dir outputs
  python tools/debug_print_snapshot.py --md outputs/debug_snapshot.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ===========================
# 默认输出文件（可通过参数覆盖）
# ===========================
DEFAULT_OUTPUT_DIR = Path("outputs")

DEFAULT_FILES = {
    "Step2 Candidate Pool（候选涨停池）": "step2_candidates.csv",
    "Step3 StrengthScore 输出（强度评分）": "step3_strength.csv",
    "Final Top10 输出（最终榜单）": "predict_top10_latest.csv",
}


# ===========================
# 读文件：尽量稳
# ===========================
def _read_csv_safely(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """
    尽力读取 CSV/TSV：
    - 自动尝试 utf-8-sig / utf-8 / gbk
    - 自动尝试分隔符：,  \\t  ;
    返回 (df, msg)。df 为 None 表示失败。
    """
    if not path.exists():
        return None, f"⚠️ 文件不存在：{path}"

    if path.stat().st_size == 0:
        return None, f"⚠️ 文件为空：{path}"

    encodings = ["utf-8-sig", "utf-8", "gbk"]
    seps = [",", "\t", ";"]

    last_err: Optional[Exception] = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")

                # 有些文件读出来只有 1 列且列名像整行，换分隔符再试
                if df.shape[1] == 1 and df.columns.size == 1:
                    col0 = str(df.columns[0])
                    if any(s in col0 for s in [",", "\t", ";"]):
                        continue

                return df, f"✅ 读取成功：{path.name}（encoding={enc}, sep={repr(sep)}）"
            except Exception as e:
                last_err = e

    return None, f"❌ 读取失败：{path}（最后错误：{last_err}）"


# ===========================
# 打印：可读、可控
# ===========================
def _maybe_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果有常见列，就做一个轻量排序，便于肉眼对比。
    规则：
    - rank/排名/ts_code/股票代码：升序
    - score/prob/StrengthScore 等：降序
    """
    if df is None or df.empty:
        return df

    # 你可能会遇到的常见列
    prefer_cols = [
        "rank", "排名",
        "ts_code", "股票代码",
        "StrengthScore", "强度得分",
        "prob", "涨停概率",
        "_score", "score", "综合得分",
    ]

    sort_cols = [c for c in prefer_cols if c in df.columns]
    if not sort_cols:
        return df

    # 去重保持顺序
    seen = set()
    uniq_cols = []
    for c in sort_cols:
        if c not in seen:
            uniq_cols.append(c)
            seen.add(c)

    def _asc_for(col: str) -> bool:
        key = col.lower()
        # “排名 / rank / code”这类通常升序
        if col in ("rank", "排名", "ts_code", "股票代码"):
            return True
        if "rank" in key or "code" in key:
            return True
        # 其余默认按“分数/概率”降序
        return False

    ascending = [_asc_for(c) for c in uniq_cols]

    try:
        return df.sort_values(by=uniq_cols, ascending=ascending)
    except Exception:
        return df


def _print_df(title: str, df: Optional[pd.DataFrame], n: int = 10, max_colwidth: int = 32) -> str:
    """
    打印并返回同样内容的字符串（方便写入 markdown）。
    """
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append(f"📌 {title}")
    lines.append("=" * 78)

    if df is None:
        lines.append("（无数据）")
        out = "\n".join(lines)
        print(out)
        return out

    if df.empty:
        lines.append("（DataFrame 为空）")
        out = "\n".join(lines)
        print(out)
        return out

    df2 = _maybe_sort(df.copy())

    # 截断超长字段，避免刷屏
    def _truncate(x):
        s = "" if pd.isna(x) else str(x)
        if len(s) > max_colwidth:
            return s[: max_colwidth - 1] + "…"
        return s

    try:
        for col in df2.columns:
            if df2[col].dtype == "object":
                df2[col] = df2[col].map(_truncate)
    except Exception:
        pass

    with pd.option_context(
        "display.max_rows", n,
        "display.max_columns", 200,
        "display.width", 220,
        "display.max_colwidth", max_colwidth,
    ):
        head_txt = df2.head(n).to_string(index=False)
        lines.append(head_txt)
        lines.append(f"\n✅ 总行数: {len(df2)}")
        lines.append(f"✅ 列数: {len(df2.columns)}")
        lines.append(f"✅ 列名: {list(df2.columns)}")

    out = "\n".join(lines)
    print(out)
    return out


def _to_markdown_block(title: str, df: Optional[pd.DataFrame], n: int = 10) -> str:
    """
    输出一个 markdown 片段（表格形式，适合放到 md 文件里）
    """
    md = []
    md.append(f"\n## {title}\n")

    if df is None:
        md.append("> （无数据）\n")
        return "".join(md)

    if df.empty:
        md.append("> （DataFrame 为空）\n")
        return "".join(md)

    df2 = _maybe_sort(df.copy()).head(n)
    try:
        md.append(df2.to_markdown(index=False))
        md.append("\n")
        md.append(f"\n- 总行数：{len(df)}\n")
        md.append(f"- 列名：{list(df.columns)}\n")
    except Exception:
        # to_markdown 依赖 tabulate；如果没装，回退成纯文本
        md.append("```text\n")
        md.append(df2.to_string(index=False))
        md.append("\n```\n")
        md.append(f"\n- 总行数：{len(df)}\n")
        md.append(f"- 列名：{list(df.columns)}\n")

    return "".join(md)


# ===========================
# 主入口
# ===========================
def main() -> int:
    parser = argparse.ArgumentParser(description="Top10 系统旁路调试打印器（稳定版）")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录（默认 outputs）")
    parser.add_argument("--n", type=int, default=10, help="每个表打印前 N 行（默认 10）")
    parser.add_argument("--max-colwidth", type=int, default=32, help="字符串列最大显示宽度（默认 32）")
    parser.add_argument("--md", type=str, default="", help="可选：写入 markdown 文件路径")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    n = max(1, int(args.n))
    max_colwidth = max(8, int(args.max_colwidth))

    print("\n✅ Top10 系统旁路调试打印器启动...\n")
    print(f"📁 输出目录：{out_dir.resolve()}")
    print(f"🔎 每表显示行数：{n}\n")

    md_parts = []
    if args.md:
        md_parts.append(f"# Top10 Debug Snapshot\n\n- 输出目录：`{out_dir}`\n- 每表显示：前 {n} 行\n")

    for title, fname in DEFAULT_FILES.items():
        path = out_dir / fname
        df, msg = _read_csv_safely(path)
        print(msg)

        _print_df(title, df, n=n, max_colwidth=max_colwidth)

        if args.md:
            md_parts.append(_to_markdown_block(title, df, n=n))

    if args.md:
        md_path = Path(args.md)
        try:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text("".join(md_parts), encoding="utf-8")
            print(f"\n📝 已写入 Markdown：{md_path.resolve()}\n")
        except Exception as e:
            print(f"\n⚠️ 写入 Markdown 失败：{e}\n")

    print("\n✅ 旁路调试结束。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
