#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/debug_print_snapshot.py

✅ 旁路调试工具（稳定版）：
- 不修改任何 step 文件
- 单独运行即可打印 pipeline 各阶段输出
- 兼容文件缺失 / 空文件 / 编码异常 / 分隔符异常（csv/tsv）
- ✅ 兼容 Markdown 表格（.md）与 JSON（.json）自动解析打印

用法：
  python tools/debug_print_snapshot.py
可选：
  python tools/debug_print_snapshot.py --n 20
  python tools/debug_print_snapshot.py --dir outputs
  python tools/debug_print_snapshot.py --md outputs/debug_snapshot.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd


# ===========================
# 默认输出目录
# ===========================
DEFAULT_OUTPUT_DIR = Path("outputs")


# ===========================
# 默认文件映射（支持多个候选）
# - 每个 title 对应多个候选文件名/通配符
# - 会自动选择“第一个存在的”，或通配符匹配到的“最新文件”
# ===========================
DEFAULT_FILES: Dict[str, List[str]] = {
    "Step2 Candidate Pool（候选涨停池）": [
        "step2_candidates.csv",
        "step2_candidates.tsv",
        "step2_candidates.md",
        "step2_candidates.json",
    ],
    "Step3 StrengthScore 输出（强度评分）": [
        "step3_strength.csv",
        "step3_strength.tsv",
        "step3_strength.md",
        "step3_strength.json",
    ],
    "Final Top10 输出（最终榜单）": [
        # 你之前设想的
        "predict_top10_latest.csv",
        "predict_top10_latest.md",
        "predict_top10_latest.json",
        # repo 里常见的
        "latest.md",
        "predict_top10_*.csv",
        "predict_top10_*.md",
        "predict_top10_*.json",
    ],
}


# ===========================
# 工具：选择一个可用文件（支持通配符取最新）
# ===========================
def _resolve_existing_file(out_dir: Path, patterns: List[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            candidates.extend(sorted(out_dir.glob(p)))
        else:
            candidates.append(out_dir / p)

    existing = [x for x in candidates if x.exists() and x.is_file() and x.stat().st_size > 0]
    if not existing:
        return None

    # 取最新：优先按 mtime，其次按名字（predict_top10_YYYYMMDD.xxx 也能正常排序）
    existing.sort(key=lambda x: (x.stat().st_mtime, x.name))
    return existing[-1]


# ===========================
# 读 CSV/TSV：尽量稳
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
# 读 Markdown 表格：尽量稳
# ===========================
def _read_markdown_table(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """
    解析 markdown 里的 pipe 表格：
      | a | b |
      |---|---|
      | 1 | 2 |
    找到第一张表就读。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return None, f"❌ 读取失败（md）：{path}（错误：{e}）"

    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    table_start = -1

    # 找到表头 + 分隔线
    for i in range(len(lines) - 1):
        if "|" in lines[i] and "|" in lines[i + 1]:
            sep_line = lines[i + 1].strip()
            if re.match(r"^\s*\|?\s*[-: ]+\|\s*[-:| ]+\|?\s*$", sep_line):
                table_start = i
                break

    if table_start < 0:
        # 没表格就当普通文本
        return None, f"⚠️ 未发现 Markdown 表格：{path.name}"

    # 收集连续的表格行
    tbl = []
    j = table_start
    while j < len(lines) and ("|" in lines[j]) and lines[j].strip():
        tbl.append(lines[j].strip())
        j += 1

    if len(tbl) < 3:
        return None, f"⚠️ Markdown 表格不完整：{path.name}"

    header = tbl[0]
    sep = tbl[1]
    rows = tbl[2:]

    # 清洗：去掉首尾 |
    def split_row(r: str) -> List[str]:
        r = r.strip()
        if r.startswith("|"):
            r = r[1:]
        if r.endswith("|"):
            r = r[:-1]
        return [c.strip() for c in r.split("|")]

    cols = split_row(header)
    data = [split_row(r) for r in rows]

    # 对齐列数
    max_len = max(len(cols), *(len(r) for r in data))
    cols = cols + [""] * (max_len - len(cols))
    fixed = []
    for r in data:
        if len(r) < max_len:
            r = r + [""] * (max_len - len(r))
        fixed.append(r[:max_len])

    try:
        df = pd.DataFrame(fixed, columns=cols)
        # 去掉空列名（很常见）
        df.columns = [c if c else f"col_{i}" for i, c in enumerate(df.columns)]
        return df, f"✅ 读取成功：{path.name}（markdown table）"
    except Exception as e:
        return None, f"❌ 解析 Markdown 表格失败：{path.name}（错误：{e}）"


# ===========================
# 读 JSON：尽量稳
# ===========================
def _read_json_safely(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)

        if isinstance(obj, list):
            df = pd.json_normalize(obj)
            return df, f"✅ 读取成功：{path.name}（json list）"

        if isinstance(obj, dict):
            # 常见：dict 里包了 topN / full / data 等
            for key in ["topN", "topn", "data", "items", "rows", "result", "full"]:
                if key in obj and isinstance(obj[key], list):
                    df = pd.json_normalize(obj[key])
                    return df, f"✅ 读取成功：{path.name}（json dict[{key}]）"
            # 兜底：把 dict 展平
            df = pd.json_normalize(obj)
            return df, f"✅ 读取成功：{path.name}（json dict）"

        return None, f"⚠️ JSON 结构不支持（不是 list/dict）：{path.name}"
    except Exception as e:
        return None, f"❌ 读取失败（json）：{path.name}（错误：{e}）"


# ===========================
# 总入口：按后缀自动选择读取方式
# ===========================
def _read_any_safely(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    suf = path.suffix.lower()
    if suf in [".csv", ".tsv"]:
        return _read_csv_safely(path)
    if suf in [".md", ".markdown"]:
        return _read_markdown_table(path)
    if suf == ".json":
        return _read_json_safely(path)
    # 兜底：尝试按 csv
    return _read_csv_safely(path)


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

    prefer_cols = [
        "rank", "排名",
        "ts_code", "股票代码",
        "StrengthScore", "强度得分", "强度评分",
        "prob", "涨停概率",
        "_score", "score", "综合得分",
    ]

    sort_cols = [c for c in prefer_cols if c in df.columns]
    if not sort_cols:
        return df

    seen = set()
    uniq_cols = []
    for c in sort_cols:
        if c not in seen:
            uniq_cols.append(c)
            seen.add(c)

    def _asc_for(col: str) -> bool:
        key = col.lower()
        if col in ("rank", "排名", "ts_code", "股票代码"):
            return True
        if "rank" in key or "code" in key:
            return True
        return False  # 其余默认降序

    ascending = [_asc_for(c) for c in uniq_cols]

    try:
        return df.sort_values(by=uniq_cols, ascending=ascending)
    except Exception:
        return df


def _print_df(title: str, df: Optional[pd.DataFrame], n: int = 10, max_colwidth: int = 32) -> str:
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append(f"📌 {title}")
    lines.append("=" * 78)

    if df is None:
        lines.append("（无数据 / 解析失败 / 文件不存在）")
        out = "\n".join(lines)
        print(out)
        return out

    if df.empty:
        lines.append("（DataFrame 为空）")
        out = "\n".join(lines)
        print(out)
        return out

    df2 = _maybe_sort(df.copy())

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
    md = []
    md.append(f"\n## {title}\n")

    if df is None:
        md.append("> （无数据 / 解析失败 / 文件不存在）\n")
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
    print(f"🔎 每表显示行数：{n}")
    print(f"📏 最大列宽：{max_colwidth}\n")

    if not out_dir.exists():
        print(f"⚠️ 输出目录不存在：{out_dir}")
        print("   你可以先跑一次 pipeline 生成 outputs，或指定正确 --dir\n")

    md_parts: List[str] = []
    if args.md:
        md_parts.append(f"# Top10 Debug Snapshot\n\n- 输出目录：`{out_dir}`\n- 每表显示：前 {n} 行\n")

    for title, patterns in DEFAULT_FILES.items():
        resolved = _resolve_existing_file(out_dir, patterns)
        if resolved is None:
            print(f"⚠️ 未找到可用文件：{title}（尝试过：{patterns}）")
            df = None
            _print_df(title, df, n=n, max_colwidth=max_colwidth)
            if args.md:
                md_parts.append(_to_markdown_block(title, df, n=n))
            continue

        df, msg = _read_any_safely(resolved)
        print(msg + f"  -> {resolved}")
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
