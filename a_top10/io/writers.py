from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn):
    """
    兼容新版 Step6 输出结构：
      topn = {"topN": DataFrame, "full": DataFrame}
    """

    outdir = Path(settings.io.outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # ① 解析 Step6 输出（兼容旧版本）
    # -------------------------------------------------
    topN_df = None
    full_df = None

    if isinstance(topn, dict):
        topN_df = topn.get("topN")
        full_df = topn.get("full")
    else:
        # fallback：旧版本只有 DataFrame
        topN_df = topn

    # -------------------------------------------------
    # ② JSON 输出
    # -------------------------------------------------
    payload = {
        "trade_date": trade_date,
        "gate": gate,
        "topN": [] if topN_df is None else topN_df.to_dict(orient="records"),
        "learn": learn,
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------------------------------------------------
    # ③ Markdown 输出：Top10 + 可选 full 排序
    # -------------------------------------------------
    md_path = outdir / f"predict_top10_{trade_date}.md"

    lines = [f"# Top10 Prediction ({trade_date})\n"]

    if topN_df is None or len(topN_df) == 0:
        lines.append("\n⚠️ Gate 未通过或候选为空。\n")
    else:
        lines.append("## 🏆 TopN\n")
        lines.append(topN_df.to_markdown(index=False))
        lines.append("\n")

    # full 排序（可选）
    if full_df is not None and len(full_df) > 0:
        lines.append("## 📊 Full Ranking\n")
        # 只展示关键字段，避免太长
        cols = [
            c for c in ["ts_code", "name", "_score", "_prob", "_strength", "_theme"]
            if c in full_df.columns
        ]
        lines.append(full_df[cols].head(50).to_markdown(index=False))
        lines.append("\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # -------------------------------------------------
    # ④ latest.md
    # -------------------------------------------------
    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"✅ Outputs written: {md_path}")
