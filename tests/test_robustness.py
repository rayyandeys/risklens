import unittest
import numpy as np
from threadpoolctl import threadpool_limits

from test_baseline import fixture
from risklens_core.features import FEATURES
from risklens_core.robustness import ABLATIONS, ablated_features, make_ablation_pipeline, paired_bootstrap_deltas


class RobustnessTests(unittest.TestCase):
    def test_declared_ablation_feature_sets(self):
        self.assertEqual(len(ABLATIONS), 5)
        self.assertEqual(ablated_features("full"), FEATURES)
        self.assertNotIn("credit_risk_score", ablated_features("no_credit_risk_score"))
        self.assertNotIn("proposed_credit_limit", ablated_features("no_proposed_credit_limit"))
        self.assertNotIn("customer_age", ablated_features("no_customer_age"))
        with self.assertRaises(KeyError):
            ablated_features("invented")

    def test_ablation_pipeline_trains_and_predicts(self):
        frame = fixture(500)
        frame.loc[::4, "prev_address_months_count"] = -1
        frame.loc[::9, "intended_balcon_amount"] = -2
        with threadpool_limits(limits=2):
            pipe = make_ablation_pipeline("no_credit_risk_or_limit", max_iter=4).fit(frame[FEATURES], frame.fraud_bool)
            output = pipe.predict_proba(frame[FEATURES].iloc[:12])[:, 1]
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(((0 <= output) & (output <= 1)).all())
        self.assertNotIn("credit_risk_score", pipe.named_steps["clean_select"].features)
        self.assertNotIn("proposed_credit_limit", pipe.named_steps["clean_select"].features)

    def test_bootstrap_is_paired_reproducible_and_zero_for_identical_scores(self):
        rng = np.random.default_rng(9)
        y = np.array([0, 1] * 100)
        scores = rng.random(200)
        ids = np.arange(1000, 1200)
        first = paired_bootstrap_deltas(y, scores, scores, ids, repeats=100, seed=7)
        second = paired_bootstrap_deltas(y, scores, scores, ids, repeats=100, seed=7)
        self.assertEqual(first, second)
        for metric in ["recall_delta", "average_precision_delta"]:
            self.assertEqual(first[metric]["mean"], 0.0)
            self.assertEqual(first[metric]["ci_lower"], 0.0)
            self.assertEqual(first[metric]["ci_upper"], 0.0)

    def test_bootstrap_rejects_bad_arguments(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([.1, .9, .2, .8])
        ids = np.arange(4)
        with self.assertRaises(ValueError):
            paired_bootstrap_deltas(y, scores, scores[:-1], ids, repeats=100)
        with self.assertRaises(ValueError):
            paired_bootstrap_deltas(y, scores, scores, ids, repeats=10)
