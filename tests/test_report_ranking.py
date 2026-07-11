from __future__ import annotations

import unittest

import pandas as pd

from a_top10.io.report_postprocess import _rank_day


class PublishedRankingMetricsTests(unittest.TestCase):
    def test_published_rank_wins_over_probability_resort(self) -> None:
        day = pd.DataFrame(
            {
                "ts_code": ["A", "B", "C"],
                "rank": [1, 2, 3],
                "Probability": [0.61, 0.95, 0.80],
            }
        )
        ranked = _rank_day(day)
        self.assertEqual(ranked["ts_code"].tolist(), ["A", "B", "C"])

    def test_legacy_rows_fall_back_to_probability(self) -> None:
        day = pd.DataFrame(
            {
                "ts_code": ["A", "B"],
                "Probability": [0.4, 0.7],
            }
        )
        ranked = _rank_day(day)
        self.assertEqual(ranked["ts_code"].tolist(), ["B", "A"])


if __name__ == "__main__":
    unittest.main()
