from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from a_top10.io.writers import _merge_feature_history, _write_prediction_ledger


class PredictionLedgerTests(unittest.TestCase):
    def test_replay_cannot_replace_live_prediction(self) -> None:
        old = pd.DataFrame(
            [{
                "trade_date": "20260710",
                "ts_code": "600000.SH",
                "run_mode": "auto_daily",
                "Probability": 0.7,
                "rank": 1,
            }]
        )
        replay = pd.DataFrame(
            [{
                "trade_date": "20260710",
                "ts_code": "600000.SH",
                "run_mode": "replay",
                "Probability": 0.2,
                "rank": 9,
                "y_limit_hit": 1,
            }]
        )
        merged = _merge_feature_history(old, replay)
        self.assertEqual(float(merged.iloc[0]["Probability"]), 0.7)
        self.assertEqual(int(merged.iloc[0]["rank"]), 1)
        self.assertEqual(int(merged.iloc[0]["y_limit_hit"]), 1)

    def test_live_and_replay_use_separate_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frame = pd.DataFrame(
                [{"trade_date": "20260710", "ts_code": "600000.SH", "run_id": "1"}]
            )
            _write_prediction_ledger(directory, frame, "auto_daily")
            _write_prediction_ledger(directory, frame, "replay")
            self.assertTrue((directory / "prediction_ledger.csv").exists())
            self.assertTrue((directory / "prediction_ledger_replay.csv").exists())


if __name__ == "__main__":
    unittest.main()
