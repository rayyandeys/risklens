import unittest
import numpy as np
import pandas as pd

from risklens_core.governance import (
    age_groups, compact_group_deltas, error_partition, error_summary,
    subgroup_metrics, top_capacity_mask,
)


class GovernanceTests(unittest.TestCase):
    def test_top_capacity_mask_respects_floor_and_row_order_ties(self):
        scores = np.array([0.9, 0.8, 0.8, 0.1])
        ids = np.array([10, 11, 12, 13], dtype=np.int64)
        first = top_capacity_mask(scores, ids, capacity=0.5, seed=42)
        perm = np.array([2, 0, 3, 1])
        second_perm = top_capacity_mask(scores[perm], ids[perm], capacity=0.5, seed=42)
        selected_first = set(ids[first].tolist())
        selected_second = set(ids[perm][second_perm].tolist())
        self.assertEqual(2, int(first.sum()))
        self.assertEqual(selected_first, selected_second)

    def test_age_groups_are_coarse_and_missing_safe(self):
        got = age_groups([20, 49, 50, 80, np.nan]).tolist()
        self.assertEqual(["age_lt_50", "age_lt_50", "age_ge_50", "age_ge_50", "missing"], got)

    def test_subgroup_metrics_use_global_selection(self):
        y = np.array([1, 0, 1, 0, 0])
        selected = np.array([True, True, False, False, True])
        groups = np.array(["a", "a", "b", "b", "b"], dtype=object)
        rows = {r["group"]: r for r in subgroup_metrics(y, selected, groups)}
        self.assertEqual(2, rows["a"]["reviewed"])
        self.assertEqual(1, rows["a"]["tp"])
        self.assertAlmostEqual(1.0, rows["a"]["recall"])
        self.assertEqual(1, rows["b"]["reviewed"])
        self.assertEqual(0, rows["b"]["tp"])
        self.assertAlmostEqual(0.0, rows["b"]["recall"])

    def test_error_partition_and_summary_counts(self):
        y = np.array([1, 0, 1, 0])
        selected = np.array([True, True, False, False])
        scores = np.array([0.9, 0.8, 0.3, 0.1])
        labels = error_partition(y, selected).tolist()
        self.assertEqual(["true_positive", "false_positive", "false_negative", "true_negative"], labels)
        summary = {r["error_class"]: r for r in error_summary(y, scores, selected)}
        self.assertTrue(all(summary[k]["rows"] == 1 for k in summary))

    def test_group_deltas_require_same_groups(self):
        ref = [{"group":"a", "selection_rate":0.1, "recall":0.2, "false_positive_rate":0.01}]
        cand = [{"group":"a", "selection_rate":0.2, "recall":0.25, "false_positive_rate":0.02}]
        delta = compact_group_deltas(cand, ref)[0]
        self.assertAlmostEqual(0.1, delta["selection_rate_delta"])
        self.assertAlmostEqual(0.05, delta["recall_delta"])
        with self.assertRaises(ValueError):
            compact_group_deltas([{"group":"b", "selection_rate":0, "recall":0, "false_positive_rate":0}], ref)


if __name__ == "__main__":
    unittest.main()
