import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from a_top10.steps import step5_ml_probability as step5


class FiveFeatureModel:
    n_features_in_ = 5

    def __init__(self):
        self.predict_called = False

    def predict_proba(self, X):
        self.predict_called = True
        raise AssertionError("contract rejection must happen before prediction")


class ValidFeatureModel:
    n_features_in_ = len(step5.FEATURE_CONTRACT)
    feature_names_in_ = np.asarray(step5.FEATURE_CONTRACT, dtype=object)

    def __init__(self):
        self.predict_called = False
        self.step5_model_manifest_ = {
            "contract_schema_version": step5.MODEL_CONTRACT_SCHEMA_VERSION,
            "feature_schema_version": step5.MODEL_FEATURE_SCHEMA_VERSION,
            "features": list(step5.FEATURE_CONTRACT),
            "probability_is_calibrated": True,
        }

    def predict_proba(self, X):
        self.predict_called = True
        positive = np.linspace(0.25, 0.75, len(X))
        return np.column_stack([1.0 - positive, positive])


class BrokenValidFeatureModel(ValidFeatureModel):
    def predict_proba(self, X):
        self.predict_called = True
        raise ValueError("synthetic inference failure")


class ValidCoreFeatureModel:
    n_features_in_ = len(step5.CORE_FEATURE_CONTRACT)
    feature_names_in_ = np.asarray(step5.CORE_FEATURE_CONTRACT, dtype=object)

    def __init__(self):
        self.step5_model_manifest_ = {
            "contract_schema_version": step5.MODEL_CONTRACT_SCHEMA_VERSION,
            "feature_schema_version": step5.CORE_FEATURE_SCHEMA_VERSION,
            "features": list(step5.CORE_FEATURE_CONTRACT),
            "probability_is_calibrated": True,
        }

    def predict_proba(self, X):
        positive = np.full(len(X), 0.35)
        return np.column_stack([1.0 - positive, positive])


class Step5ModelContractTests(unittest.TestCase):
    def test_rejects_five_feature_model_before_prediction(self):
        model = FiveFeatureModel()
        index = pd.RangeIndex(2)
        X = pd.DataFrame(np.zeros((2, len(step5.FEATURE_CONTRACT))), columns=step5.FEATURE_CONTRACT)

        result = step5._predict_with_model(model, X, index)

        self.assertEqual(result.contract.status, "feature_count_mismatch")
        self.assertEqual(result.contract.expected_feature_count, 21)
        self.assertEqual(result.contract.actual_feature_count, 5)
        self.assertIn("expected_feature_count=21", result.contract.reason)
        self.assertIn("actual_feature_count=5", result.contract.reason)
        self.assertEqual(result.prediction_status, "rejected")
        self.assertTrue(result.values.isna().all())
        self.assertFalse(model.predict_called)

    def test_accepts_ordered_valid_model_and_preserves_calibration_manifest(self):
        model = ValidFeatureModel()
        index = pd.RangeIndex(2)
        X = pd.DataFrame(np.zeros((2, len(step5.FEATURE_CONTRACT))), columns=step5.FEATURE_CONTRACT)

        result = step5._predict_with_model(model, X, index)

        self.assertEqual(result.contract.status, "valid")
        self.assertEqual(result.prediction_status, "ok")
        self.assertTrue(result.contract.probability_is_calibrated)
        np.testing.assert_allclose(result.values.to_numpy(), np.asarray([0.25, 0.75]))
        self.assertTrue(model.predict_called)

    def test_prediction_exception_is_exposed_in_audit_result(self):
        model = BrokenValidFeatureModel()
        index = pd.RangeIndex(1)
        X = pd.DataFrame(np.zeros((1, len(step5.FEATURE_CONTRACT))), columns=step5.FEATURE_CONTRACT)

        result = step5._predict_with_model(model, X, index)

        self.assertEqual(result.contract.status, "valid")
        self.assertEqual(result.prediction_status, "error")
        self.assertEqual(result.prediction_error, "ValueError: synthetic inference failure")
        self.assertTrue(result.values.isna().all())
        self.assertTrue(model.predict_called)

    def test_versioned_core_model_is_accepted_without_guessing_legacy_model(self):
        model = ValidCoreFeatureModel()
        features = step5._model_feature_contract(model)
        self.assertEqual(features, step5.CORE_FEATURE_CONTRACT)
        X = pd.DataFrame(np.zeros((2, len(features))), columns=features)
        result = step5._predict_with_model(model, X, X.index, feature_names=features)
        self.assertEqual(result.contract.status, "valid")
        self.assertTrue(result.contract.probability_is_calibrated)
        np.testing.assert_allclose(result.values.to_numpy(), [0.35, 0.35])

    def test_training_falls_back_to_core_when_enhancement_has_no_overlap(self):
        frame = pd.DataFrame(
            {
                **{c: [1.0, 2.0] for c in step5.CORE_FEATURE_CONTRACT},
                "intraday_available": [0, 0],
                "auction_available": [1, 1],
            }
        )
        features, meta = step5._select_training_feature_contract(frame, 0.85)
        self.assertEqual(features, step5.CORE_FEATURE_CONTRACT)
        self.assertEqual(meta["feature_mode"], "core")

    def test_time_calibration_uses_trailing_trade_dates(self):
        rows = 60
        X = pd.DataFrame(
            np.linspace(0.0, 1.0, rows * len(step5.CORE_FEATURE_CONTRACT)).reshape(
                rows, len(step5.CORE_FEATURE_CONTRACT)
            ),
            columns=step5.CORE_FEATURE_CONTRACT,
        )
        y = np.asarray([0, 1, 0, 1] * 15)
        dates = [f"202601{day:02d}" for day in range(1, 16) for _ in range(4)]
        model, meta = step5._fit_time_calibrated_model(
            step5.LogisticRegression(max_iter=100), X, y, dates
        )
        self.assertIsNotNone(model)
        self.assertEqual(meta["reason"], "ok")
        self.assertEqual(meta["split"], "trailing_trade_dates")
        self.assertGreater(meta["calibration_rows"], 0)
        self.assertIn("validation", meta)

    def test_rule_fallback_is_explicitly_uncalibrated_and_auditable(self):
        input_df = pd.DataFrame(
            {
                "trade_date": ["20260710"],
                "ts_code": ["600000.SH"],
                "name": ["测试股份"],
                "StrengthScore": [72.0],
                "ThemeBoost": [0.2],
                "seal_amount": [10_000_000.0],
                "open_times": [0.0],
                "turnover_rate": [8.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(io=SimpleNamespace(outputs_dir=Path(temp_dir)))
            with mock.patch.object(step5, "load_lr", return_value=FiveFeatureModel()), mock.patch.object(
                step5, "load_hgb", return_value=FiveFeatureModel()
            ), mock.patch.object(step5, "load_lgbm", return_value=FiveFeatureModel()):
                output = step5.run_step5(input_df, s=settings)

        row = output.iloc[0]
        self.assertEqual(row["_prob_src"], "fallback_rule")
        self.assertEqual(row["probability_semantics"], "rank_score_uncalibrated")
        self.assertFalse(bool(row["probability_is_calibrated"]))
        self.assertEqual(row["model_contract_status"], "fallback_uncalibrated")
        self.assertEqual(row["lr_model_contract_status"], "feature_count_mismatch")
        self.assertEqual(row["lgbm_model_contract_status"], "feature_count_mismatch")
        self.assertIn("actual_feature_count=5", row["model_contract_reason"])
        self.assertAlmostEqual(float(row["rank_score"]), float(row["Probability"]))

    def test_calibrated_hgb_sets_canonical_probability_column(self):
        input_df = pd.DataFrame(
            {
                "trade_date": ["20260710"],
                "ts_code": ["600000.SH"],
                "StrengthScore": [72.0],
                "ThemeBoost": [0.2],
                "seal_amount": [10_000_000.0],
                "open_times": [0.0],
                "turnover_rate": [8.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(io=SimpleNamespace(outputs_dir=Path(temp_dir)))
            with mock.patch.object(step5, "load_lr", return_value=FiveFeatureModel()), mock.patch.object(
                step5, "load_hgb", return_value=ValidCoreFeatureModel()
            ), mock.patch.object(step5, "load_lgbm", return_value=FiveFeatureModel()):
                output = step5.run_step5(input_df, s=settings)

        row = output.iloc[0]
        self.assertEqual(row["_prob_src"], "hgb")
        self.assertTrue(bool(row["probability_is_calibrated"]))
        self.assertEqual(row["probability_semantics"], "calibrated_probability")
        self.assertAlmostEqual(float(row["p_limit_up_calibrated"]), 0.35)


if __name__ == "__main__":
    unittest.main()
