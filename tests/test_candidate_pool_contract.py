from __future__ import annotations

import unittest

import pandas as pd

from a_top10.steps.step2_candidate_pool import _filter_close_limit_up


class CandidatePoolContractTests(unittest.TestCase):
    def test_only_verified_close_limit_up_rows_survive(self) -> None:
        frame = pd.DataFrame(
            {
                "ts_code": ["A", "B", "C"],
                "limit_type": ["U", "U", "D"],
                "close": [11.0, 10.8, 9.0],
                "up_limit": [11.0, 11.0, 11.0],
            }
        )
        out, stats = _filter_close_limit_up(frame, enabled=True)
        self.assertEqual(out["ts_code"].tolist(), ["A"])
        self.assertEqual(stats["dropped_rows"], 2)

    def test_unverifiable_source_fails_closed(self) -> None:
        frame = pd.DataFrame({"ts_code": ["A"]})
        with self.assertRaisesRegex(RuntimeError, "cannot prove"):
            _filter_close_limit_up(frame, enabled=True)


if __name__ == "__main__":
    unittest.main()
