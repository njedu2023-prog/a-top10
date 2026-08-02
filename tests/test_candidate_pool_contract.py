from __future__ import annotations

import unittest

import pandas as pd

from a_top10.steps.step2_candidate_pool import (
    _filter_close_limit_up,
    _filter_standard_10pct_limit,
)


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

    def test_only_standard_10pct_price_limit_regime_survives(self) -> None:
        frame = pd.DataFrame(
            {
                "ts_code": [
                    "600001.SH",
                    "600002.SH",
                    "300001.SZ",
                    "688001.SH",
                    "920001.BJ",
                ],
                "up_limit": [11.0, 10.5, 12.0, 12.0, 13.0],
                "down_limit": [9.0, 9.5, 8.0, 8.0, 7.0],
            }
        )

        out, stats = _filter_standard_10pct_limit(frame)

        self.assertEqual(out["ts_code"].tolist(), ["600001.SH"])
        self.assertEqual(stats["kept_rows"], 1)
        self.assertEqual(stats["dropped_rows"], 4)
        self.assertEqual(
            stats["regime_counts"],
            {"5pct": 1, "10pct": 1, "20pct": 2, "30pct": 1, "other_or_unknown": 0},
        )

    def test_tick_rounded_10pct_limit_is_not_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "ts_code": ["600001.SH"],
                "up_limit": [1.11],
                "down_limit": [0.91],
            }
        )

        out, stats = _filter_standard_10pct_limit(frame)

        self.assertEqual(out["ts_code"].tolist(), ["600001.SH"])
        self.assertEqual(stats["regime_counts"]["10pct"], 1)

    def test_board_code_fallback_is_fail_closed(self) -> None:
        frame = pd.DataFrame(
            {
                "ts_code": [
                    "600001.SH",
                    "000001.SZ",
                    "300001.SZ",
                    "688001.SH",
                    "920001.BJ",
                    "UNKNOWN",
                ]
            }
        )

        out, stats = _filter_standard_10pct_limit(frame)

        self.assertEqual(out["ts_code"].tolist(), ["600001.SH", "000001.SZ"])
        self.assertEqual(stats["regime_counts"]["other_or_unknown"], 1)
        self.assertEqual(stats["source_counts"]["unknown"], 1)


if __name__ == "__main__":
    unittest.main()
