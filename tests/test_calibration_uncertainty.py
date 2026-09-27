import unittest

import numpy as np
import pandas as pd

from risklens_core.calibration_uncertainty import (
    IsotonicScoreCalibrator,
    SigmoidScoreCalibrator,
    calibration_bins,
    calibration_metrics,
    fit_score_calibrators,
    month_label_bootstrap_positions,
    queue_overlap,
    select_calibration_method,
    summarize_stability,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        self.scores = np.array([0.02, 0.08, 0.18, 0.35, 0.30, 0.55, 0.75, 0.92], dtype=float)
        self.row_ids = np.arange(100, 108, dtype=np.int64)

    def test_calibrators_fit_predict_and_stay_bounded(self):
        fitted = fit_score_calibrators(self.scores, self.y)
        self.assertEqual(set(fitted), {"sigmoid", "isotonic"})
        for calibrator in fitted.values():
            p = calibrator.predict(self.scores)
            self.assertEqual(p.shape, self.scores.shape)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(SigmoidScoreCalibrator().fit(self.scores, self.y).parameters()["slope_on_raw_logit"], 0)
        self.assertGreaterEqual(IsotonicScoreCalibrator().fit(self.scores, self.y).parameters()["threshold_count"], 2)

    def test_reliability_and_metric_validation(self):
        result = calibration_bins(self.y, self.scores, bins=4)
        self.assertEqual(sum(row["count"] for row in result["bins"]), len(self.y))
        self.assertGreaterEqual(result["ece"], 0)
        report = calibration_metrics(self.y, self.scores, self.row_ids, bins=4)
        self.assertIn("brier_score", report)
        self.assertIn("operational", report)
        with self.assertRaises(ValueError):
            calibration_bins(np.zeros(len(self.y), dtype=int), self.scores)

    def test_selection_rule_is_brier_then_ece(self):
        reports = {
            "raw": {"brier_score": 0.10, "ece": 0.04},
            "sigmoid": {"brier_score": 0.09, "ece": 0.05},
            "isotonic": {"brier_score": 0.09, "ece": 0.03},
        }
        self.assertEqual(select_calibration_method(reports), "isotonic")
        reports["sigmoid"] = {"brier_score": 0.08, "ece": 0.10}
        self.assertEqual(select_calibration_method(reports), "sigmoid")

    def test_monotone_sigmoid_keeps_queue_membership(self):
        calibrator = SigmoidScoreCalibrator().fit(self.scores, self.y)
        calibrated = calibrator.predict(self.scores)
        overlap = queue_overlap(self.scores, calibrated, self.row_ids, capacity=0.5, seed=42)
        self.assertEqual(overlap["reference_retained_fraction"], 1.0)
        self.assertEqual(overlap["jaccard"], 1.0)


class StabilityTests(unittest.TestCase):
    def test_stratified_bootstrap_preserves_month_label_counts_and_seed(self):
        frame = pd.DataFrame({
            "month": [0, 0, 0, 0, 1, 1, 1, 1],
            "fraud_bool": [0, 0, 1, 1, 0, 0, 1, 1],
            "x": np.arange(8),
        })
        a = month_label_bootstrap_positions(frame, seed=7)
        b = month_label_bootstrap_positions(frame, seed=7)
        self.assertTrue(np.array_equal(a, b))
        sampled = frame.iloc[a]
        expected = frame.groupby(["month", "fraud_bool"]).size().sort_index()
        actual = sampled.groupby(["month", "fraud_bool"]).size().sort_index()
        pd.testing.assert_series_equal(actual, expected)

    def test_stability_summary_detects_boundary_switching(self):
        ref = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=float)
        ids = np.arange(6, dtype=np.int64)
        ensemble = np.array([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.9, 0.75, 0.78, 0.6, 0.5, 0.4],
            [0.9, 0.82, 0.72, 0.6, 0.5, 0.4],
        ])
        summary, cases = summarize_stability(ref, ensemble, ids, capacity=0.5, seed=42)
        self.assertEqual(summary["repeats"], 3)
        self.assertEqual(summary["reference_selected_cases"], 3)
        self.assertEqual(len(cases), 6)
        self.assertTrue(((cases["selection_frequency"] >= 0) & (cases["selection_frequency"] <= 1)).all())
        self.assertIn("queue_jaccard_vs_reference", summary)

    def test_stability_input_guards(self):
        ref = np.array([0.8, 0.2])
        ids = np.array([1, 2])
        with self.assertRaises(ValueError):
            summarize_stability(ref, np.array([[0.8, 0.2]]), ids)
        bad = pd.DataFrame({"month": [0, 0], "fraud_bool": [0, 0]})
        with self.assertRaises(ValueError):
            month_label_bootstrap_positions(bad, seed=1)


if __name__ == "__main__":
    unittest.main()
