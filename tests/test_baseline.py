import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from risklens_core.features import SCHEMA, FEATURES, CATEGORICAL, BINARY, NUMERIC, FeatureCleaner
from risklens_core.baseline import make_pipeline, load_development, sha256_file, diagnostics
from risklens_core.metrics import evaluate


def fixture(n=160):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({c: rng.uniform(0, 1, n) for c in SCHEMA})
    for c in CATEGORICAL:
        frame[c] = np.where(np.arange(n) % 2, "A", "B")
    for c in BINARY:
        frame[c] = np.arange(n) % 2
    frame["month"] = np.arange(n) % 8
    frame["fraud_bool"] = (np.arange(n) // 8) % 2
    return frame


class BaselineTests(unittest.TestCase):
    def test_column_specific_negative_handling(self):
        frame = fixture()
        frame.loc[0, ["credit_risk_score", "velocity_6h", "intended_balcon_amount",
                      "prev_address_months_count"]] = [-1, -5, -2.6, -1]
        result = FeatureCleaner().transform(frame)
        self.assertEqual(result.loc[0, "credit_risk_score"], -1)
        self.assertEqual(result.loc[0, "velocity_6h"], -5)
        self.assertTrue(pd.isna(result.loc[0, "intended_balcon_amount"]))
        self.assertEqual(result.loc[0, "prev_address_months_count__missing"], 1)

    def test_target_and_time_excluded(self):
        result = FeatureCleaner().transform(fixture())
        for c in ["fraud_bool", "month", "days_since_request", "device_fraud_count"]:
            self.assertNotIn(c, result)

    def test_nonbinary_feature_rejected(self):
        frame = fixture()
        frame.loc[0, "email_is_free"] = 3
        with self.assertRaisesRegex(ValueError, "Non-binary"):
            FeatureCleaner().transform(frame)

    def test_infinite_feature_rejected(self):
        frame = fixture()
        frame.loc[0, "income"] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            FeatureCleaner().transform(frame)

    def test_preprocessing_stays_training_fitted_and_handles_unknown_categories(self):
        frame = fixture()
        frame.loc[0, "bank_months_count"] = -1
        pipe = make_pipeline()
        pipe.fit(frame[FEATURES], frame.fraud_bool)
        preprocessing = pipe.named_steps["preprocess"]
        imputer = preprocessing.named_transformers_["numeric"].named_steps["impute"]
        column = NUMERIC.index("bank_months_count")
        self.assertAlmostEqual(imputer.statistics_[column], frame.bank_months_count.iloc[1:].median())
        before = imputer.statistics_.copy()
        unseen = frame[FEATURES].copy()
        unseen["bank_months_count"] = 999999
        unseen["payment_type"] = "UNSEEN_CATEGORY"
        probabilities = pipe.predict_proba(unseen)
        np.testing.assert_array_equal(before, imputer.statistics_)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_pipeline_serialization_preserves_predictions(self):
        frame = fixture()
        pipe = make_pipeline().fit(frame[FEATURES], frame.fraud_bool)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(pipe, path)
            np.testing.assert_allclose(pipe.predict_proba(frame), joblib.load(path).predict_proba(frame))

    def test_capacity_counts_against_hand_calculation(self):
        result = evaluate([1, 0, 1, 0], [.9, .8, .7, .1], [0, 1, 2, 3], capacities=(.5,))
        row = result["review_capacity"][0]
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (1, 1, 1, 1))
        self.assertEqual(row["recall"], .5)
        self.assertEqual(row["precision"], .5)

    def test_floor_budget_does_not_overspend(self):
        result = evaluate([1, 0, 0], [.9, .2, .1], [0, 1, 2], capacities=(.5, .01))
        self.assertEqual(result["review_capacity"][0]["reviewed"], 1)
        self.assertEqual(result["review_capacity"][1]["reviewed"], 0)
        self.assertIsNone(result["review_capacity"][1]["precision"])

    def test_tied_scores_invariant_to_row_order(self):
        y, p, ids = np.array([1, 0, 1, 0]), np.ones(4) * .5, np.array([13, 10, 16, 12])
        first = evaluate(y, p, ids, capacities=(.5,))
        second = evaluate(y[::-1], p[::-1], ids[::-1], capacities=(.5,))
        self.assertEqual(first, second)

    def test_bad_probabilities_rejected(self):
        with self.assertRaisesRegex(ValueError, "Probabilities"):
            evaluate([0, 1], [.1, np.nan], [0, 1])

    def test_cross_split_feature_duplicate_stops_training(self):
        train = fixture()
        val = train.iloc[:2].copy()
        val["month"] = 5
        with self.assertRaisesRegex(ValueError, "match training hashes"):
            diagnostics(train, val)

    def test_loading_discards_holdout_and_rejects_changed_file(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Base.csv"
            frame.to_csv(path, index=False)
            audit = {"sha256": sha256_file(path),
                     "month_rows": {str(k): int(v) for k, v in frame.month.value_counts().items()},
                     "development_target_summary": {}}
            for name, mask in [("train", frame.month <= 4), ("validation", frame.month == 5)]:
                audit["development_target_summary"][name] = {
                    "rows": int(mask.sum()), "fraud_rows": int(frame.loc[mask].fraud_bool.sum())}
            train, val, _, _ = load_development(path, audit)
            self.assertEqual(set(train.month), {0, 1, 2, 3, 4})
            self.assertEqual(set(val.month), {5})
            self.assertFalse(set(train.index) & set(val.index))
            with path.open("a") as file:
                file.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_development(path, audit)
