from __future__ import annotations

import unittest

import pandas as pd

from a_top10.steps.step6_final_topn import run_step6_final_topn
from tools.apply_limit_stage_overlay import apply_stage_score


class RankingContractTest(unittest.TestCase):
    def assert_probability_descending(self, frame: pd.DataFrame) -> None:
        probability = pd.to_numeric(frame["prob_final"], errors="raise")
        self.assertTrue(
            probability.is_monotonic_decreasing,
            f"probability inversion detected: {probability.tolist()}",
        )

    def test_step6_probability_outranks_higher_composite_score(self) -> None:
        source = pd.DataFrame(
            {
                "ts_code": ["600001.SH", "600002.SH"],
                "name": ["高概率", "高综合分"],
                "prob_final": [0.81, 0.80],
                "StrengthScore": [0.0, 100.0],
                "ThemeBoost": [0.0, 1.30],
                "stage_quality_weight": [1.0, 1.0],
                "stage_risk_weight": [0.0, 0.0],
            }
        )

        ranked = run_step6_final_topn(source)["full"]

        self.assertGreater(
            float(ranked.loc[ranked["ts_code"] == "600002.SH", "final_score_v2"].iloc[0]),
            float(ranked.loc[ranked["ts_code"] == "600001.SH", "final_score_v2"].iloc[0]),
        )
        self.assertEqual(ranked["ts_code"].tolist(), ["600001.SH", "600002.SH"])
        self.assert_probability_descending(ranked)

    def test_overlay_is_idempotent_and_does_not_double_stage_adjustment(self) -> None:
        source = pd.DataFrame(
            {
                "ts_code": ["600003.SH"],
                "name": ["阶段样本"],
                "prob_final": [0.70],
                "StrengthScore": [50.0],
                "ThemeBoost": [0.50],
                "stage_quality_weight": [1.10],
                "stage_risk_weight": [0.0],
            }
        )
        step6 = run_step6_final_topn(source)["full"]

        once = apply_stage_score(step6)
        twice = apply_stage_score(once)
        expected = round(
            float(once.loc[0, "final_score_pre_stage"])
            + float(once.loc[0, "stage_adjustment"]),
            6,
        )

        self.assertAlmostEqual(float(once.loc[0, "stage_adjustment"]), 0.01, places=6)
        self.assertAlmostEqual(float(once.loc[0, "final_score_v2"]), expected, places=6)
        self.assertAlmostEqual(
            float(twice.loc[0, "final_score_v2"]),
            float(once.loc[0, "final_score_v2"]),
            places=6,
        )

    def test_overlay_uses_canonical_probability_not_composite_or_compat_field(self) -> None:
        source = pd.DataFrame(
            {
                "ts_code": ["600010.SH", "600020.SH"],
                "name": ["规范概率第一", "综合分第一"],
                "prob_final": [0.90, 0.80],
                "Probability": [0.10, 0.99],
                "prob": [0.10, 0.99],
                "final_score_pre_stage": [0.10, 0.99],
                "final_score_v2": [0.10, 0.99],
                "stage_adjustment": [0.0, 0.0],
                "stage_quality_weight": [1.0, 1.0],
                "stage_risk_weight": [0.0, 0.0],
            }
        )

        ranked = apply_stage_score(source)

        self.assertEqual(ranked["ts_code"].tolist(), ["600010.SH", "600020.SH"])
        self.assertEqual(ranked["rank"].tolist(), [1, 2])
        self.assertEqual(ranked["Probability"].tolist(), [0.90, 0.80])
        self.assertEqual(ranked["prob"].tolist(), [0.90, 0.80])
        self.assert_probability_descending(ranked)

    def test_equal_probabilities_have_stable_tie_break(self) -> None:
        source = pd.DataFrame(
            {
                "ts_code": ["600102.SH", "600101.SH", "600103.SH"],
                "name": ["乙", "甲", "丙"],
                "prob_final": [0.75, 0.75, 0.75],
                "StrengthScore": [99.0, 1.0, 50.0],
                "ThemeBoost": [1.30, 0.0, 0.50],
                "stage_quality_weight": [1.0, 1.0, 1.0],
                "stage_risk_weight": [0.0, 0.0, 0.0],
            }
        )

        first = run_step6_final_topn(source)["full"]["ts_code"].tolist()
        second = run_step6_final_topn(source.iloc[::-1].reset_index(drop=True))["full"]["ts_code"].tolist()

        self.assertEqual(first, ["600101.SH", "600102.SH", "600103.SH"])
        self.assertEqual(first, second)

    def test_overlay_preserves_probability_audit_contract(self) -> None:
        source = pd.DataFrame(
            {
                "ts_code": ["600001.SH"],
                "Probability": [0.32],
                "p_limit_up_calibrated": [0.32],
                "probability_is_calibrated": [True],
                "probability_semantics": ["calibrated_probability"],
                "model_schema_version": ["step5_core_features_v1"],
                "_prob_src": ["hgb"],
                "stage_quality_weight": [1.0],
                "stage_risk_weight": [0.0],
            }
        )
        out = apply_stage_score(source)
        self.assertEqual(float(out.loc[0, "p_limit_up_calibrated"]), 0.32)
        self.assertTrue(bool(out.loc[0, "probability_is_calibrated"]))
        self.assertEqual(out.loc[0, "_prob_src"], "hgb")


if __name__ == "__main__":
    unittest.main()
