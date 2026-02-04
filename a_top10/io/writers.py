from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Optional, Sequence, Mapping

import pandas as pd


def _df_to_md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    """
    安全地把 DataFrame 转成 Markdown 表格：
    - 优先 df.to_markdown（依赖 tabulate）
    - 若环境缺 tabulate，则降级为手写 pipe table（不会报错）
    ✅ 表头输出为中文
    """
    if df is None or df.empty:
        return ""

    if cols is not None:
        use_cols = [c for c in cols if c in df.columns]
        if use_cols:
            df = df[use_cols].copy()

    # ✅ 表头中文映射（只影响输出，不影响 DataFrame）
    col_map = {
        "rank": "排名",
        "ts_code": "股票代码",
        "name": "名称",
        "score": "综合得分",
        "prob": "涨停概率",
        "StrengthScore": "强度得分",
        "ThemeBoost": "题材加成",
        "board": "板块",
    }

    # ✅ 输出前替换表头显示名
    df = df.rename(columns=col_map)

    try:
        return df.to_markdown(index=False)
    except Exception:
        d = df.copy().fillna("")
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


def _pick_first_not_none(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """
    ✅ 关键：不要用 `a or b or c` 来选 DataFrame（会触发 DataFrame.__bool__ -> ValueError）
    """
    for k in keys:
        if k in d:
            v = d.get(k)
            if v is not None:
                return v
    return None


def _to_df(x: Any) -> Optional[pd.DataFrame]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    try:
        return pd.DataFrame(x)
    except Exception:
        return None


def write_outputs(settings, trade_date: str, ctx, gate, topn, learn) -> None:
    """
    ✅ writers.py 最终稳定版

    兼容 Step6 输出：
      1) dict: {"topn"/"topN"/"TopN": DataFrame, "full": DataFrame}
      2) 旧版: 直接 DataFrame

    输出：
      - predict_top10_{trade_date}.json
      - predict_top10_{trade_date}.md
      - latest.md（覆盖）
    """
    outdir = Path(settings.io.outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # ① 解析 Step6 输出（稳定写法，不触发 DataFrame truth-value）
    # -------------------------------------------------
    topN_df: Optional[pd.DataFrame] = None
    full_df: Optional[pd.DataFrame] = None

    if isinstance(topn, dict):
        topN_df = _pick_first_not_none(topn, ["topN", "topn", "TopN", "top"])
        full_df = topn.get("full") if "full" in topn else None
    else:
        topN_df = topn

    topN_df = _to_df(topN_df)
    full_df = _to_df(full_df)

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
    # ③ Markdown 输出（Top10 + Full）
    # -------------------------------------------------
    md_path = outdir / f"predict_top10_{trade_date}.md"
    lines = [f"# Top10 Prediction ({trade_date})\n"]

    # --- Top10 区 ---
    if topN_df is None or topN_df.empty:
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
        top_cols = ["rank", "ts_code", "name", "score", "prob", "StrengthScore", "ThemeBoost", "board"]
        lines.append(_df_to_md_table(topN_df, cols=top_cols))
        lines.append("\n")

    # --- Full 排序区（只展示前 50，防止 md 过大） ---
    if full_df is not None and not full_df.empty:
        lines.append("## 📊 Full Ranking (All Candidates After Step6)\n")

        full_sorted = full_df.copy()

        # 优先按 Step6 的内部列排序
        if "_score" in full_sorted.columns:
            full_sorted = full_sorted.sort_values(
                by=["_score", "_prob"] if "_prob" in full_sorted.columns else ["_score"],
                ascending=False,
            )
        elif "score" in full_sorted.columns:
            full_sorted = full_sorted.sort_values(
                by=["score", "prob"] if "prob" in full_sorted.columns else ["score"],
                ascending=False,
            )
        elif "prob" in full_sorted.columns:
            full_sorted = full_sorted.sort_values(by=["prob"], ascending=False)

        full_sorted = full_sorted.head(50)

        display_cols = ["rank", "ts_code", "name", "score", "prob", "StrengthScore", "ThemeBoost", "board"]
        lines.append(_df_to_md_table(full_sorted, cols=display_cols))
        lines.append("\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # -------------------------------------------------
    # ④ latest.md（覆盖最新预测）
    # -------------------------------------------------
    latest = outdir / "latest.md"
    latest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"✅ Outputs written: {md_path}")
