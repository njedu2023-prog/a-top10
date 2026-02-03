from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputPaths:
    root: str = "outputs"

    def ensure(self) -> Path:
        p = Path(self.root)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def md_path(self, trade_date: str) -> Path:
        return self.ensure() / f"predict_top10_{trade_date}.md"

    def json_path(self, trade_date: str) -> Path:
        return self.ensure() / f"predict_top10_{trade_date}.json"

    def latest_md_path(self) -> Path:
        return self.ensure() / "latest.md"
