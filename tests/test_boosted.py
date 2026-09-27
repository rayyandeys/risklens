from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from test_baseline import fixture
from risklens_core.boosted import make_boosted_pipeline, validate_reference
from risklens_core.features import FEATURES, CONTRACT_VERSION


class BoostedTests(unittest.TestCase):
    def test_native_categories_missing_values_and_roundtrip(self):
        frame = fixture(800)
        frame.loc[::3, "prev_address_months_count"] = -1
        frame.loc[::7, "payment_type"] = np.nan
        with threadpool_limits(limits=2):
            pipe = make_boosted_pipeline(max_iter=5).fit(frame[FEATURES], frame.fraud_bool)
            self.assertEqual(int(pipe.named_steps["model"].is_categorical_.sum()), 5)
            self.assertFalse(pipe.named_steps["model"].do_early_stopping_)
            unseen = frame[FEATURES].iloc[:8].copy()
            unseen["payment_type"] = "UNKNOWN"
            output = pipe.predict_proba(unseen)
            self.assertTrue(np.isfinite(output).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "tree.joblib"
                joblib.dump(pipe, path)
                np.testing.assert_allclose(output, joblib.load(path).predict_proba(unseen))

    def reference(self):
        return {"status": "complete", "test_evaluated": False, "dataset_sha256": "hash",
                "feature_contract": CONTRACT_VERSION, "features": FEATURES,
                "split": {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]},
                "models": {n: {"converged": True} for n in ["logistic_unweighted", "logistic_balanced"]}}

    def test_compatible_reference_accepted(self):
        validate_reference(self.reference(), "hash")

    def test_different_fingerprint_rejected(self):
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_reference(self.reference(), "different")

    def test_test_evaluated_reference_rejected(self):
        ref = self.reference()
        ref["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "development-only"):
            validate_reference(ref, "hash")

    def test_changed_features_or_split_rejected(self):
        for field, value in [("features", FEATURES[:-1]), ("split", {})]:
            with self.subTest(field=field):
                ref = deepcopy(self.reference())
                ref[field] = value
                with self.assertRaises(ValueError):
                    validate_reference(ref, "hash")
