from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn):
    """
    最小输出器：
    outputs/predict_top10_YYYYMMDD.md
    outputs/predict_top10_YYYYMMDD.json
    outputs/latest.md
    """

    outdir = Path(settings.io.outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    payload = {
        "trade_date": trade_date,
        "gate": gate,
        "topn": None if topn is None else topn.to_dict(orient="records"),
    }

    json_path = outdir / f"predict_top10_{trade_date}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Markdown ---
    md_path = outdir / f"predict_top10_{trade_date}.md"

    if topn is None or topn.empty:
        body = "⚠️ Gate 未通过，Top10为空。\n"
    else:
        body = topn.to_markdown(index=False)

    md_path.write_text(
        f"# Top10 Prediction ({trade_date})\n\n{body}\n",
        encoding="utf-8",
    )

    # --- latest ---
    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print("✅ Outputs written:", md_path)
