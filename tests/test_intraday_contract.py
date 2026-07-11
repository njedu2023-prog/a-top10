import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from a_top10.intraday_features import (
    build_intraday_debug_summary,
    merge_intraday_to_candidates,
    normalize_score_series,
    prepare_auction_features,
    prepare_intraday_features,
)
from a_top10.steps.step0_input_layer import step0_build_universe
from tools.check_intraday_refine import ContractValidationError, validate_feature_contract


class ScoreNormalizationTests(unittest.TestCase):
    def test_normalizes_0_100_without_saturating(self):
        raw = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "limitup_quality_score": [25.0, 80.0],
                "intraday_risk_score": [10.0, 90.0],
                "late_withdraw_score": [0.0, 70.0],
                "reseal_acceptance_score": [20.0, 60.0],
                "limitup_path_score": [30.0, 90.0],
                "open_board_count": [0, 2],
            }
        )

        out = prepare_intraday_features(raw, {})

        self.assertAlmostEqual(float(out.loc[0, "limitup_quality_score"]), 0.25)
        self.assertAlmostEqual(float(out.loc[1, "limitup_quality_score"]), 0.80)
        self.assertAlmostEqual(float(out.loc[1, "intraday_risk_score_raw"]), 0.90)
        self.assertLess(float(out["limitup_quality_score"].max()), 1.0)

    def test_preserves_0_1_and_rejects_out_of_contract_values(self):
        normalized = normalize_score_series(pd.Series([0.2, 0.8]))
        self.assertEqual(normalized.tolist(), [0.2, 0.8])

        raw = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "limitup_quality_score": [120.0, -1.0],
            }
        )
        out = prepare_intraday_features(raw, {})
        self.assertEqual(out["limitup_quality_is_default"].tolist(), [1, 1])
        self.assertEqual(out["limitup_quality_score"].tolist(), [0.55, 0.55])


class CandidateCoverageTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["A", "B"],
                "StrengthScore": [80.0, 70.0],
            }
        )

    def test_coverage_requires_code_intersection_and_nondegenerate_features(self):
        intraday = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "limitup_quality_score": [20.0, 80.0],
                "intraday_risk_score": [70.0, 30.0],
                "open_board_count": [0, 2],
            }
        )
        auction = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "auction_strength_score": [35.0, 75.0],
            }
        )

        merged = merge_intraday_to_candidates(self.candidates, intraday, auction, {})
        debug = build_intraday_debug_summary(merged, merged, {"intraday_rows": 2, "auction_rows": 2}, "20260710")

        self.assertEqual(merged["intraday_available"].tolist(), [1, 1])
        self.assertEqual(merged["auction_available"].tolist(), [1, 1])
        self.assertAlmostEqual(float(merged.loc[0, "auction_strength_score"]), 0.35)
        self.assertEqual(merged["auction_fake_strength_score"].tolist(), [0.0, 0.0])
        self.assertEqual(debug["code_overlap_count"], 2)
        self.assertEqual(debug["valid_feature_count"], 2)
        self.assertIn("limitup_quality_score", debug["nondegenerate_features"])
        self.assertIn("auction_strength_score", debug["auction_nondegenerate_features"])

    def test_zero_code_intersection_never_reports_coverage(self):
        intraday = pd.DataFrame(
            {
                "ts_code": ["600001.SH", "600002.SH"],
                "limitup_quality_score": [20.0, 80.0],
            }
        )
        merged = merge_intraday_to_candidates(self.candidates, intraday, pd.DataFrame(), {})
        debug = build_intraday_debug_summary(merged, merged, {"intraday_rows": 2}, "20260710")

        self.assertEqual(int(merged["intraday_available"].sum()), 0)
        self.assertEqual(debug["code_overlap_count"], 0)
        self.assertEqual(debug["matched_rate"], 0.0)
        self.assertEqual(set(merged["intraday_status"]), {"missing_stock"})

    def test_constant_candidate_features_are_not_counted_as_valid_coverage(self):
        intraday = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "limitup_quality_score": [50.0, 50.0],
                "intraday_risk_score": [20.0, 20.0],
                "open_board_count": [0, 0],
            }
        )
        merged = merge_intraday_to_candidates(self.candidates, intraday, pd.DataFrame(), {})

        self.assertEqual(int(merged["intraday_available"].sum()), 0)
        self.assertEqual(set(merged["intraday_status"]), {"degenerate_features"})
        self.assertTrue((merged["intraday_matched_key"] != "").all())

    def test_raw_auction_remains_compatible_but_is_not_false_coverage(self):
        raw = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "vol": [1000, 2000],
                "price": [10.0, 20.0],
                "amount": [10000.0, 40000.0],
            }
        )
        prepared = prepare_auction_features(raw, {})
        merged = merge_intraday_to_candidates(self.candidates, pd.DataFrame(), raw, {})

        self.assertEqual(int(prepared["auction_available"].sum()), 0)
        self.assertEqual(int(merged["auction_available"].sum()), 0)
        self.assertEqual(set(merged["auction_data_status"]), {"invalid_features"})


class Step0AuctionSourceTests(unittest.TestCase):
    class Repo:
        name = "a-share-top3-data"

        def __init__(self, snapshot: Path):
            self.snapshot = snapshot

        def snapshot_dir(self, trade_date: str) -> Path:
            return self.snapshot

    def test_prefers_auction_features_and_keeps_legacy_context_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            pd.DataFrame(
                {"ts_code": ["000001.SZ"], "name": ["A"], "close": [10.0], "pct_chg": [10.0]}
            ).to_csv(snapshot / "daily.csv", index=False)
            pd.DataFrame(
                {"ts_code": ["000001.SZ"], "auction_strength_score": [72.0]}
            ).to_csv(snapshot / "auction_features.csv", index=False)
            pd.DataFrame(
                {"ts_code": ["000001.SZ"], "vol": [1000], "price": [10.0], "amount": [10000.0]}
            ).to_csv(snapshot / "stk_auction.csv", index=False)
            settings = SimpleNamespace(
                data_repo=self.Repo(snapshot),
                intraday=SimpleNamespace(
                    enabled=True,
                    require_intraday_features=False,
                    require_stk_auction=False,
                    missing_policy="neutral",
                ),
            )

            ctx = step0_build_universe(settings, "20260710")

            self.assertEqual(ctx["auction_source"], "auction_features.csv")
            self.assertIn("auction_strength_score", ctx["stk_auction"].columns)
            self.assertIs(ctx["stk_auction"], ctx["auction_df"])
            self.assertIn("vol", ctx["raw_stk_auction"].columns)
            self.assertEqual(ctx["debug"]["intraday_input"]["auction_source"], "auction_features.csv")

            (snapshot / "auction_features.csv").unlink()
            raw_ctx = step0_build_universe(settings, "20260710")
            self.assertEqual(raw_ctx["auction_source"], "stk_auction.csv")
            self.assertIn("vol", raw_ctx["stk_auction"].columns)


class CheckerContractTests(unittest.TestCase):
    def _debug(self):
        return {
            "candidate_count": 2,
            "intraday_rows": 2,
            "code_overlap_count": 2,
            "code_overlap_rate": 1.0,
            "matched_count": 2,
            "matched_rate": 1.0,
            "valid_feature_count": 2,
            "valid_feature_rate": 1.0,
            "nondegenerate_features": ["limitup_quality_score"],
            "feature_distributions": {
                "limitup_quality_score": {
                    "sample_count": 2,
                    "unique_count": 2,
                    "upper_saturation_rate": 0.0,
                }
            },
            "auction_rows": 0,
            "auction_source": "",
            "auction_code_overlap_count": 0,
            "auction_code_overlap_rate": 0.0,
            "auction_valid_count": 0,
            "auction_valid_rate": 0.0,
            "auction_nondegenerate_features": [],
            "auction_feature_distributions": {},
        }

    def test_checker_accepts_consistent_contract(self):
        audit = pd.DataFrame({"intraday_available": [1, 1]})
        result = validate_feature_contract(audit, self._debug())
        self.assertEqual(result["valid_feature_rate"], 1.0)

    def test_checker_explains_zero_overlap(self):
        audit = pd.DataFrame({"intraday_available": [0, 0]})
        debug = self._debug()
        for key in ["code_overlap_count", "code_overlap_rate", "matched_count", "matched_rate", "valid_feature_count", "valid_feature_rate"]:
            debug[key] = 0
        debug["nondegenerate_features"] = []
        debug["feature_distributions"] = {}

        with self.assertRaisesRegex(ContractValidationError, "intraday_code_overlap_zero"):
            validate_feature_contract(audit, debug, require_intraday=True)

        result = validate_feature_contract(audit, debug, require_intraday=False)
        self.assertIn("intraday_code_overlap_zero", result["warnings"][0])

    def test_checker_explains_upper_saturation(self):
        audit = pd.DataFrame({"intraday_available": [1, 1]})
        debug = self._debug()
        debug["feature_distributions"]["limitup_quality_score"]["upper_saturation_rate"] = 0.99

        with self.assertRaisesRegex(ContractValidationError, "intraday_feature_upper_saturated"):
            validate_feature_contract(audit, debug, require_intraday=True)

    def test_checker_explains_degenerate_coverage(self):
        audit = pd.DataFrame({"intraday_available": [0, 0]})
        debug = self._debug()
        for key in ["matched_count", "matched_rate", "valid_feature_count", "valid_feature_rate"]:
            debug[key] = 0
        debug["nondegenerate_features"] = []
        debug["feature_distributions"] = {
            "limitup_quality_score": {
                "sample_count": 2,
                "unique_count": 1,
                "upper_saturation_rate": 0.0,
            }
        }

        with self.assertRaisesRegex(ContractValidationError, "intraday_valid_nondegenerate_coverage_zero"):
            validate_feature_contract(audit, debug, require_intraday=True)


if __name__ == "__main__":
    unittest.main()
