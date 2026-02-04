from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Optional, Sequence

import pandas as pd


def _df_to_md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    """
    安全地把 DataFrame 转成 Markdown 表格：
    - 优先 df.to_markdown（需要 tabulate）
    - 若环境缺 tabulate，则降级为手写 pipe table（不会报错）
    """
    if df is None or df.empty:
        return ""

    if cols is not None:
        use_cols = [c for c in cols if c in df.columns]
        if use_cols:
            df = df[use_cols].copy()

    # 尝试 pandas 内置 to_markdown（依赖 tabulate）
    try:
        return df.to_markdown(index=False)
    except Exception:
        # 降级：手写 Markdown pipe table（简单稳定）
        d = df.copy()
        d = d.fillna("")
        headers = list(d.columns)

        def esc(x: Any) -> str:
            s = str(x)
            s = s.replace("\n", " ").replace("\r", " ")
            s = s.replace("|", "\\|")
            return s

        lines = []
        lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
        return "\n".join(lines)


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn):
    """
    适配新版 Step6 输出结构：
      topn = {
        "topN": DataFrame,
        "full": DataFrame
      }

    字段匹配 step6_final_topn.py 新版字段：
      ["ts_code","name","score","prob","board","StrengthScore","ThemeBoost"]
    """
    outdir = Path(settings.io.outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # ① 解析 Step6 输出
    # -------------------------------------------------
    topN_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        topN_df = topn.get("topN")
        full_df = topn.get("full")
    else:
        # fallback 兼容旧版本
        topN_df = topn

    # 兜底：保证是 DataFrame 或 None
    if topN_df is not None and not isinstance(topN_df, pd.DataFrame):
        topN_df = pd.DataFrame(topN_df)
    if full_df is not None and not isinstance(full_df, pd.DataFrame):
        full_df = pd.DataFrame(full_df)

    # -------------------------------------------------
    # ② JSON 输出
    # -------------------------------------------------
    payload: Dict[str, Any] = {
        "trade_date": trade_date,
        "gate": gate,
        "topN": [] if topN_df is None else topN_df.to_dict(orient="records"),
        "full": [] if full_df is None else full_df.to_dict(orient="records"),
        "learn": learn,
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # ③ Markdown 输出（Top10 + full 排序）
    # -------------------------------------------------
    md_path = outdir / f"predict_top10_{trade_date}.md"
    lines = [f"# Top10 Prediction ({trade_date})\n"]

    # --- TopN 区 ---
    if topN_df is None or topN_df.empty:
        # gate 可能是 dict，尽量把原因写出来
        reason = ""
        try:
            if isinstance(gate, dict):
                r = gate.get("reason") or gate.get("msg") or ""
                if r:
                    reason = f"（{r}）"
        except Exception:
            pass
        lines.append(f"⚠️ Gate 未通过，Top10 为空。{reason}\n")
    else:
        lines.append("## 🏆 Top10 (Final Selection)\n")
        top_cols = ["ts_code", "name", "score", "prob", "StrengthScore", "ThemeBoost", "board"]
        lines.append(_df_to_md_table(topN_df, cols=top_cols))
        lines.append("\n")

    # --- Full 排序区（只展示核心字段） ---
    if full_df is not None and not full_df.empty:
        lines.append("## 📊 Full Ranking (All Candidates After Step6)\n")

        # 若存在 score/prob，做一个更符合直觉的排序
        full_sorted = full_df.copy()
        if "score" in full_sorted.columns:
            full_sorted = full_sorted.sort_values(by=["score"], ascending=False)
        elif "prob" in full_sorted.columns:
            full_sorted = full_sorted.sort_values(by=["prob"], ascending=False)

        display_cols = ["ts_code", "name", "score", "prob", "StrengthScore", "ThemeBoost", "board"]
        lines.append(_df_to_md_table(full_sorted, cols=display_cols))
        lines.append("\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # -------------------------------------------------
    # ④ latest.md（覆盖最新预测）
    # -------------------------------------------------
    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"✅ Outputs written: {md_path}")
