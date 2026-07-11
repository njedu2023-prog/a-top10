from __future__ import annotations

import unittest

import pandas as pd

from a_top10.steps.step3_strength_score import _robust_rank01


class StrengthDirectionContractTests(unittest.TestCase):
    def test_high_is_good_assigns_highest_score_to_largest_value(self) -> None:
        scores = _robust_rank01(pd.Series([1.0, 2.0, 3.0]), higher_is_better=True)
        self.assertEqual(int(scores.idxmax()), 2)
        self.assertGreater(float(scores.iloc[2]), float(scores.iloc[0]))

    def test_low_is_good_assigns_highest_score_to_smallest_value(self) -> None:
        scores = _robust_rank01(pd.Series([0.0, 1.0, 4.0]), higher_is_better=False)
        self.assertEqual(int(scores.idxmax()), 0)
        self.assertGreater(float(scores.iloc[0]), float(scores.iloc[2]))


if __name__ == "__main__":
    unittest.main()
