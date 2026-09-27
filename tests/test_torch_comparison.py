import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from risklens_core.features import FEATURES
from risklens_core.torch_comparison import (
    TorchMLPConfig,
    fit_transform_preprocessor,
    load_torch_bundle,
    make_preprocessor,
    parameter_count,
    predict_scores,
    require_torch,
    save_torch_bundle,
    train_mlp,
)


def frame(rows=120, seed=1):
    rng = np.random.default_rng(seed)
    data = {}
    categorical_values = {
        "payment_type": ["AA", "AB"],
        "employment_status": ["CA", "CB"],
        "housing_status": ["BA", "BB"],
        "source": ["INTERNET", "TELEAPP"],
        "device_os": ["windows", "linux"],
    }
    binary = {"email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards", "foreign_request", "keep_alive_session"}
    for column in FEATURES:
        if column in categorical_values:
            data[column] = rng.choice(categorical_values[column], size=rows)
        elif column in binary:
            data[column] = rng.integers(0, 2, size=rows)
        else:
            data[column] = rng.normal(1.0, 0.5, size=rows)
    # Exercise all canonical missing sentinels.
    for column in ["prev_address_months_count", "current_address_months_count", "bank_months_count", "session_length_in_minutes", "device_distinct_emails_8w"]:
        data[column][0] = -1
    data["intended_balcon_amount"][1] = -2
    return pd.DataFrame(data)


class TorchComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_torch()

    def test_preprocessor_is_train_fitted_and_handles_unknown_categories(self):
        train = frame(80, 1)
        val = frame(30, 2)
        val.loc[0, "payment_type"] = "UNSEEN"
        preprocessor, x_train, x_val = fit_transform_preprocessor(train, val)
        self.assertEqual(x_train.shape[1], x_val.shape[1])
        self.assertTrue(np.isfinite(x_train).all())
        self.assertTrue(np.isfinite(x_val).all())
        self.assertGreater(x_train.shape[1], len(FEATURES))
        self.assertIsNotNone(preprocessor)

    def test_training_predicts_valid_scores(self):
        train = frame(160, 3)
        val = frame(40, 4)
        _, x_train, x_val = fit_transform_preprocessor(train, val)
        y = np.array(([0] * 140) + ([1] * 20), dtype=int)
        config = TorchMLPConfig(hidden_dims=(16, 8), dropout=0.0, epochs=2, batch_size=32, seed=7)
        model, history, metadata = train_mlp(x_train, y, config=config, device="cpu", threads=1)
        scores = predict_scores(model, x_val, device="cpu", batch_size=16)
        self.assertEqual(scores.shape, (len(val),))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        self.assertEqual(len(history), 2)
        self.assertIsNone(metadata["positive_class_weight"])
        self.assertGreater(parameter_count(model), 0)

    def test_serialization_round_trip_preserves_scores(self):
        train = frame(120, 5)
        val = frame(25, 6)
        _, x_train, x_val = fit_transform_preprocessor(train, val)
        y = np.array(([0] * 100) + ([1] * 20), dtype=int)
        config = TorchMLPConfig(hidden_dims=(12,), dropout=0.0, epochs=1, batch_size=64, seed=9)
        model, _, _ = train_mlp(x_train, y, config=config, device="cpu", threads=1)
        before = predict_scores(model, x_val)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_torch_bundle(path, model, input_dim=x_train.shape[1], config=config, metadata={"x": 1})
            loaded, loaded_config, metadata = load_torch_bundle(path)
            after = predict_scores(loaded, x_val)
        self.assertEqual(loaded_config.hidden_dims, config.hidden_dims)
        self.assertEqual(metadata, {"x": 1})
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-7)

    def test_fixed_seed_reproduces_cpu_training(self):
        train = frame(96, 7)
        _, x_train, _ = fit_transform_preprocessor(train, train.iloc[:20].copy())
        y = np.array(([0] * 80) + ([1] * 16), dtype=int)
        config = TorchMLPConfig(hidden_dims=(10,), dropout=0.0, epochs=2, batch_size=24, seed=11)
        first, _, _ = train_mlp(x_train, y, config=config, device="cpu", threads=1)
        second, _, _ = train_mlp(x_train, y, config=config, device="cpu", threads=1)
        a = predict_scores(first, x_train[:20])
        b = predict_scores(second, x_train[:20])
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)

    def test_invalid_labels_and_weight_mode_rejected(self):
        train = frame(20, 8)
        _, x_train, _ = fit_transform_preprocessor(train, train.copy())
        with self.assertRaises(ValueError):
            train_mlp(x_train, np.zeros(20, dtype=int), config=TorchMLPConfig(epochs=1, batch_size=8), threads=1)
        with self.assertRaises(ValueError):
            train_mlp(
                x_train,
                np.array(([0] * 10) + ([1] * 10)),
                config=TorchMLPConfig(epochs=1, batch_size=8, class_weighting="bad"),
                threads=1,
            )


if __name__ == "__main__":
    unittest.main()
