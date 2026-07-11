from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from a_top10.config import DataRepo, a_share_calendar_source, next_a_share_trading_day
from a_top10.steps import step7_self_learning as step7


class StrictTradingDayTests(unittest.TestCase):
    def test_authoritative_xshg_calendar_is_active(self) -> None:
        self.assertEqual(a_share_calendar_source(), "exchange_calendars:XSHG")

    def test_holiday_fallback_resolves_exact_next_a_share_day(self) -> None:
        self.assertEqual(next_a_share_trading_day("20260430"), "20260506")

    def test_data_repo_calendar_api_remains_compatible(self) -> None:
        repo = DataRepo("_warehouse", "a-share-top3-data", "data/raw")
        previous, following = repo.prev_next_trade_date("20260430")
        self.assertEqual(previous, "20260429")
        self.assertEqual(following, "20260506")

    def test_snapshot_lookup_never_skips_missing_exact_t1(self) -> None:
        resolved = step7._next_snapshot_after(
            "20260407",
            snapshot_dates=["20260407", "20260409"],
            upper_bound="20260409",
            next_trade_date_resolver=lambda _: "20260408",
        )
        self.assertEqual(resolved, "")

    def test_missing_exact_t1_invalidates_stale_t2_label(self) -> None:
        history = pd.DataFrame({
            "trade_date": ["20260407"],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
            "verify_date": ["20260409"],
            "is_sample_mature": [1],
            "mature_reason": ["next_trade_truth_ready"],
            "y_limit_hit": [1.0],
            "y_next_ret": [0.1],
        })
        warnings: list[str] = []

        result = step7._apply_maturity_and_labels(
            history,
            snapshot_dates=["20260407", "20260409"],
            upper_bound="20260409",
            warnings=warnings,
            next_trade_date_resolver=lambda _: "20260408",
        )

        row = result.iloc[0]
        self.assertEqual(row["verify_date"], "20260408")
        self.assertEqual(row["is_sample_mature"], 0)
        self.assertEqual(row["mature_reason"], "next_trade_snapshot_missing")
        self.assertTrue(pd.isna(row["y_limit_hit"]))
        self.assertTrue(pd.isna(row["y_next_ret"]))
        self.assertIn("expected_verify_date=20260408", warnings[0])

    def test_exact_t1_snapshot_is_labeled(self) -> None:
        history = pd.DataFrame({
            "trade_date": ["20260407"],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
        })
        limit_df = pd.DataFrame({"ts_code": ["000001.SZ"]})
        daily_df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [11.0]})

        with (
            patch.object(step7, "_read_limit_list_snapshot", return_value=limit_df),
            patch.object(step7, "_read_daily_snapshot", return_value=daily_df),
        ):
            result = step7._apply_maturity_and_labels(
                history,
                snapshot_dates=["20260408", "20260409"],
                upper_bound="20260409",
                warnings=[],
                next_trade_date_resolver=lambda _: "20260408",
            )

        row = result.iloc[0]
        self.assertEqual(row["verify_date"], "20260408")
        self.assertEqual(row["is_sample_mature"], 1)
        self.assertEqual(row["mature_reason"], "next_trade_truth_ready")
        self.assertEqual(row["y_limit_hit"], 1.0)
        self.assertAlmostEqual(row["y_next_ret"], 0.1)


class PublishedRankingMetricTests(unittest.TestCase):
    def _history(self) -> pd.DataFrame:
        codes = [f"00000{i}.SZ" for i in range(1, 10)] + ["000010.SZ"]
        labels = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
        return pd.DataFrame({
            "trade_date": ["20260701"] * 10,
            "verify_date": ["20260702"] * 10,
            "ts_code": codes,
            "Probability": np.linspace(0.01, 0.99, 10),
            "is_sample_mature": [1] * 10,
            "y_limit_hit": labels,
        })

    def test_metrics_follow_published_rank_not_probability(self) -> None:
        history = self._history()
        with tempfile.TemporaryDirectory() as tmp:
            learning_dir = Path(tmp) / "learning"
            learning_dir.mkdir()
            pd.DataFrame({
                "rank": range(1, 11),
                "ts_code": history["ts_code"],
            }).to_csv(learning_dir / "pred_top10_20260701.csv", index=False)

            hit_df, _, latest = step7._build_hit_history(history, learning_dir, [])

        row = hit_df.iloc[0]
        self.assertEqual(row["ranking_source"], "published_file:pred_top10_20260701.csv")
        self.assertEqual(row["top1_hit"], 0)
        self.assertEqual(row["top3_hit"], 2)
        self.assertAlmostEqual(row["top3_hit_rate"], 0.6667)
        self.assertEqual(row["top5_hit"], 3)
        self.assertEqual(row["top10_hit"], 5)
        self.assertEqual(row["topn"], 10)
        self.assertEqual(row["hit"], 5)
        self.assertEqual(row["hit_rate"], 0.5)
        self.assertEqual(latest["top1_hit"], 0)

    def test_missing_published_rank_does_not_fallback_to_probability(self) -> None:
        history = self._history()
        warnings: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            learning_dir = Path(tmp) / "learning"
            learning_dir.mkdir()
            hit_df, _, latest = step7._build_hit_history(history, learning_dir, warnings)

        row = hit_df.iloc[0]
        self.assertEqual(row["note"], "published_rank_missing")
        self.assertEqual(row["topn"], 0)
        self.assertEqual(row["hit"], "")
        self.assertIsNone(latest)
        self.assertIn("published_rank_missing: trade_date=20260701", warnings)

    def test_legacy_hit_history_columns_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            pd.DataFrame([{
                "trade_date": "20260701",
                "verify_date": "20260702",
                "topn": 10,
                "hit": 3,
                "hit_rate": 0.3,
                "note": "legacy",
            }]).to_csv(path, index=False)
            result = step7._read_hit_history_csv(path)

        self.assertEqual(result.iloc[0]["topn"], 10)
        self.assertEqual(result.iloc[0]["hit"], 3)
        self.assertEqual(result.iloc[0]["hit_rate"], 0.3)
        self.assertIn("top1_hit_rate", result.columns)

    def test_aggregate_contains_all_required_cutoffs(self) -> None:
        hit_df = pd.DataFrame({
            "top1_n": [1], "top1_hit": [0],
            "top3_n": [3], "top3_hit": [2],
            "top5_n": [5], "top5_hit": [3],
            "top10_n": [10], "top10_hit": [5],
        })
        metrics = step7._aggregate_ranking_metrics(hit_df)
        self.assertEqual(set(metrics), {"Top1", "Top3", "Top5", "Top10"})
        self.assertEqual(metrics["Top3"]["sample_count"], 3)
        self.assertEqual(metrics["Top3"]["hit_count"], 2)
        self.assertAlmostEqual(metrics["Top3"]["hit_rate"], 0.6667)


class TrainingReportConsistencyTests(unittest.TestCase):
    def test_hgb_update_is_counted_as_success(self) -> None:
        result = {
            "lr": {"trained": False, "updated": False, "reason": "not_selected"},
            "hgb": {"trained": True, "updated": True, "reason": "formal_model_updated"},
            "lgbm": {"trained": False, "updated": False, "reason": "optional_unavailable"},
        }
        trained, updated, reason = step7._summarize_step5_training_result(result, False)
        self.assertTrue(trained)
        self.assertTrue(updated)
        self.assertEqual(reason, "ok_all_pass_dates_model_updated")

    def test_reason_reports_trained_but_not_updated(self) -> None:
        result = {
            "ok": True,
            "updated": False,
            "lr": {"trained": True, "updated": False, "reason": "level1_trial_training_no_formal_update"},
            "lgbm": {"trained": False, "updated": False, "reason": "lightgbm_not_installed"},
        }
        trained, updated, reason = step7._summarize_step5_training_result(result, True)
        self.assertTrue(trained)
        self.assertFalse(updated)
        self.assertIn("partial_pass_dates_trained_not_updated", reason)
        self.assertIn("level1_trial_training_no_formal_update", reason)

    def test_reason_reports_training_failure(self) -> None:
        result = {
            "ok": False,
            "updated": False,
            "lr": {"trained": False, "updated": False, "reason": "insufficient_feature_coverage"},
            "lgbm": {"trained": False, "updated": False, "reason": "lightgbm_not_installed"},
        }
        trained, updated, reason = step7._summarize_step5_training_result(result, False)
        self.assertFalse(trained)
        self.assertFalse(updated)
        self.assertTrue(reason.startswith("step5_train_not_completed:"))


if __name__ == "__main__":
    unittest.main()
