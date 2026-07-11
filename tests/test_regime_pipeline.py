from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from a_top10.main import (
    _attach_regime_features,
    _load_previous_regime_state,
    _write_regime_state,
)


class RegimePipelineTests(unittest.TestCase):
    def test_regime_is_attached_to_every_candidate(self) -> None:
        frame = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"]})
        regime = {
            "score": 0.61,
            "smooth": 0.58,
            "weight": 0.95,
            "state": "neutral",
            "inputs": {"E1": 63, "E2": 12.5, "E3": 4},
        }
        out = _attach_regime_features(frame, regime)
        self.assertEqual(out["regime_score"].tolist(), [0.61, 0.61])
        self.assertEqual(out["regime_E3"].tolist(), [4, 4])

    def test_regime_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            gate = {
                "regime": {
                    "smooth": 0.57,
                    "score": 0.6,
                    "weight": 0.94,
                    "state": "neutral",
                    "inputs": {"E1": 50},
                }
            }
            _write_regime_state(out_dir, "20260710", gate)
            state = _load_previous_regime_state(out_dir)
            self.assertEqual(state["trade_date"], "20260710")
            self.assertEqual(state["prev_smooth"], 0.57)
            json.loads((out_dir / "learning" / "regime_state_latest.json").read_text())


if __name__ == "__main__":
    unittest.main()
