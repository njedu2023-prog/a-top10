from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn):
    """
    适配新版 Step6 输出结构：
      topn = {
        "topN": DataFrame,
        "full": DataFrame
      }
    字段匹配 step6_final_topn.py 新版字段：
      ["ts_code","name","score","prob","board",
       "StrengthScore","ThemeBoost"]
    """

    outdir = Path(settings.io.outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # ① 解析 Step6 输出
    # -------------------------------------------------
    topN_df = None
    full_df = None

    if isinstance(topn, dict):
        topN_df = topn.get("topN")
        full_df = topn.get("full")
    else:
        # fallback 兼容旧版本
        topN_df = topn

    # -------------------------------------------------
    # ② JSON 输出
    # -------------------------------------------------
    payload = {
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
    if topN_df is None or len(topN_df) == 0:
        lines.append("⚠️ Gate 未通过，Top10 为空。\n")
    else:
        lines.append("## 🏆 Top10 (Final Selection)\n")
        lines.append(topN_df.to_markdown(index=False))
        lines.append("\n")

    # --- Full 排序区（只展示核心字段） ---
    if full_df is not None and len(full_df) > 0:
        lines.append("## 📊 Full Ranking (All Candidates After Step6)\n")
        display_cols = [
            c for c in [
                "ts_code", "name", "score", "prob",
                "StrengthScore", "ThemeBoost", "board"
            ] if c in full_df.columns
        ]
        lines.append(full_df[display_cols].to_markdown(index=False))
        lines.append("\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # -------------------------------------------------
    # ④ latest.md（覆盖最新预测）
    # -------------------------------------------------
    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"✅ Outputs written: {md_path}")
