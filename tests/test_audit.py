"""Small synthetic fixtures test validation behaviour, not model performance."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("audit_data", Path(__file__).parents[1] / "scripts" / "audit_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "Base.csv"
        self.header = ["fraud_bool", "month", "feature"]
        self.rows = [[label, month, "-1" if label else ""] for month in range(8) for label in (0, 1)]

    def write(self):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(self.header)
            writer.writerows(self.rows)

    def test_counts_and_holdout_exclusion(self):
        self.write()
        result = MODULE.audit(self.path)
        self.assertEqual(result["rows"], 16)
        self.assertEqual(result["development_target_summary"]["train"]["rows"], 10)
        self.assertNotIn("test", result["development_target_summary"])
        self.assertEqual(result["training_missingness_screen"]["feature"], {"null_tokens": 5, "minus_one_tokens": 5})

    def test_fractional_month_rejected(self):
        self.rows[0][1] = 0.5
        self.write()
        with self.assertRaisesRegex(ValueError, "month must be"):
            MODULE.audit(self.path)

    def test_nonbinary_label_rejected(self):
        self.rows[0][0] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, "fraud_bool must be"):
            MODULE.audit(self.path)

    def test_nan_label_rejected(self):
        self.rows[0][0] = "nan"
        self.write()
        with self.assertRaisesRegex(ValueError, "fraud_bool must be"):
            MODULE.audit(self.path)

    def test_missing_month_rejected(self):
        self.rows = self.rows[:-2]
        self.write()
        with self.assertRaisesRegex(ValueError, "absent"):
            MODULE.audit(self.path)

    def test_wrong_row_width_rejected(self):
        self.rows[0].append("extra")
        self.write()
        with self.assertRaisesRegex(ValueError, "fields"):
            MODULE.audit(self.path)

    def test_duplicate_header_rejected(self):
        self.header[2] = "month"
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            MODULE.audit(self.path)

    def test_input_hash_changes_with_content(self):
        self.write()
        before = MODULE.audit(self.path)["sha256"]
        self.rows[0][2] = "new"
        self.write()
        self.assertNotEqual(before, MODULE.audit(self.path)["sha256"])

    def test_single_class_training_rejected(self):
        for row in self.rows:
            if row[1] <= 4:
                row[0] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, "train requires both"):
            MODULE.audit(self.path)


if __name__ == "__main__":
    unittest.main()
