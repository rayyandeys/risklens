import unittest

import numpy as np
import pandas as pd

from risklens_core.features import FEATURES
from risklens_core.monitoring import (
    categorical_drift,
    feature_drift,
    numeric_drift,
    overall_status,
    score_drift,
    summarize_feature_drift,
    validate_calibration_reference,
)


def make_frame(n=200, shift=0.0):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(index=np.arange(n))
    for feature in FEATURES:
        if feature in {"payment_type", "employment_status", "housing_status", "source", "device_os"}:
            frame[feature] = "A"
        else:
            frame[feature] = rng.normal(1.0 + shift, 0.2, n)
    for feature in ["email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards", "foreign_request", "keep_alive_session"]:
        frame[feature] = rng.integers(0, 2, n)
    # Keep sentinel-sensitive fields in valid observed ranges for this fixture.
    for feature in ["prev_address_months_count", "current_address_months_count", "bank_months_count", "session_length_in_minutes", "device_distinct_emails_8w"]:
        frame[feature] = rng.integers(0, 24, n)
    frame["intended_balcon_amount"] = rng.uniform(0, 100, n)
    return frame


class MonitoringTests(unittest.TestCase):
    def test_numeric_drift_identical_is_zero(self):
        s = pd.Series(np.linspace(0, 1, 100))
        report = numeric_drift(s, s, bins=10)
        self.assertAlmostEqual(report["psi"], 0.0, places=12)
        self.assertEqual(report["severity"], "normal")

    def test_numeric_shift_escalates(self):
        ref = pd.Series(np.linspace(0, 1, 1000))
        tar = pd.Series(np.linspace(5, 6, 1000))
        report = numeric_drift(ref, tar, bins=10)
        self.assertGreaterEqual(report["psi"], 0.25)
        self.assertEqual(report["severity"], "alert")

    def test_missingness_change_is_detected(self):
        ref = pd.Series([1.0] * 100)
        tar = pd.Series([1.0] * 90 + [np.nan] * 10)
        report = numeric_drift(ref, tar, bins=5)
        self.assertEqual(report["severity"], "alert")
        self.assertAlmostEqual(report["missing_rate_delta"], 0.10)

    def test_categorical_unseen_category_rate(self):
        ref = pd.Series(["A"] * 50 + ["B"] * 50)
        tar = pd.Series(["A"] * 50 + ["C"] * 50)
        report = categorical_drift(ref, tar)
        self.assertAlmostEqual(report["unseen_category_rate"], 0.5)
        self.assertEqual(report["severity"], "alert")

    def test_full_feature_report_and_summary(self):
        ref = make_frame(250, shift=0.0)
        tar = make_frame(250, shift=0.4)
        table = feature_drift(ref, tar)
        self.assertEqual(len(table), 34)  # 28 raw inputs + 6 missing flags
        summary = summarize_feature_drift(table, top_n=5)
        self.assertEqual(len(summary["top_features"]), 5)
        self.assertIn(summary["status"], {"normal", "watch", "alert"})

    def test_score_drift_bounds_and_status(self):
        report = score_drift(np.linspace(0.0, 0.2, 1000), np.linspace(0.5, 0.8, 1000))
        self.assertEqual(report["severity"], "alert")
        with self.assertRaises(ValueError):
            score_drift([0.1, 1.2], [0.1, 0.2])

    def test_overall_status_uses_worst_signal(self):
        self.assertEqual(overall_status({"status": "watch"}, {"severity": "alert"}), "alert")
        self.assertEqual(overall_status({"status": "normal"}, {"severity": "normal"}), "normal")

    def test_calibration_reference_guard(self):
        good = {
            "status": "complete", "experiment": "calibration-uncertainty-v1",
            "dataset_sha256": "d", "reference_model_sha256": "m",
            "test_evaluated": False, "scored_month": 5,
            "calibration": {"selected_method": "raw"},
        }
        validate_calibration_reference(good, dataset_sha256="d", model_sha256="m")
        bad = dict(good)
        bad["test_evaluated"] = True
        with self.assertRaises(ValueError):
            validate_calibration_reference(bad, dataset_sha256="d", model_sha256="m")


if __name__ == "__main__":
    unittest.main()
