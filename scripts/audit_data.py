"""Audit BAF CSV input without fitting a model or reporting holdout performance.

Python 3.12+, standard library only. Run from the project root.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
SPLIT_MONTHS = {"train": list(range(5)), "validation": [5], "test": [6, 7]}


def integer_field(raw, name, line, allowed):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"CSV line {line}: {name} must be an integer") from None
    if not math.isfinite(value) or not value.is_integer() or int(value) not in allowed:
        raise ValueError(f"CSV line {line}: {name} must be one of {sorted(allowed)}")
    return int(value)


def audit(path):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Use an extracted CSV file, not a ZIP or Parquet file.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    month_rows = {month: 0 for month in range(8)}
    development = {name: {"rows": 0, "fraud_rows": 0} for name in ("train", "validation")}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty or has no header.")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate column names are not allowed.")
        if any(name != name.strip() for name in header):
            raise ValueError("Column names contain surrounding whitespace; review the export.")
        required = {"fraud_bool", "month"}
        if not required.issubset(header):
            raise ValueError(f"Missing required columns: {sorted(required - set(header))}")
        target_index, month_index = header.index("fraud_bool"), header.index("month")
        stats = {name: {"null_tokens": 0, "minus_one_tokens": 0} for name in header}
        rows = 0
        for row in reader:
            line = reader.line_num
            if len(row) != len(header):
                raise ValueError(f"CSV line {line}: expected {len(header)} fields, got {len(row)}")
            month = integer_field(row[month_index], "month", line, set(range(8)))
            label = integer_field(row[target_index], "fraud_bool", line, {0, 1})
            rows += 1
            month_rows[month] += 1
            split = "train" if month <= 4 else "validation" if month == 5 else "test"
            if split in development:
                development[split]["rows"] += 1
                development[split]["fraud_rows"] += label
            if split == "train":
                for name, value in zip(header, row):
                    value = value.strip().lower()
                    stats[name]["null_tokens"] += value in NULL_TOKENS
                    stats[name]["minus_one_tokens"] += value in {"-1", "-1.0"}
        if not rows:
            raise ValueError("CSV contains a header but no data rows.")
    absent = [month for month, count in month_rows.items() if not count]
    if absent:
        raise ValueError(f"Expected all months 0 through 7 for this protocol; absent: {absent}")
    for split, summary in development.items():
        if summary["fraud_rows"] in (0, summary["rows"]):
            raise ValueError(f"{split} requires both target classes before baseline training.")
        summary["fraud_rate"] = summary["fraud_rows"] / summary["rows"]
    return {
        "audit_version": 1,
        "dataset_file": path.name,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "columns": header,
        "month_rows": month_rows,
        "proposed_split_months": SPLIT_MONTHS,
        "development_target_summary": development,
        "training_missingness_screen": stats,
        "possible_export_index_columns": [name for name in header if not name or name.lower().startswith("unnamed:")],
        "limitations": [
            "Input identity is recorded, but this is not full BAF schema validation.",
            "Split is a RiskLens proposal pending data and datasheet review; not a claim to reproduce the published protocol.",
            "Minus-one counts are a screen only; sentinel meanings need a column-by-column datasheet review.",
            "Duplicate records, repeated entities, and feature availability at prediction time are not yet checked.",
            "Holdout target distribution and performance are intentionally not reported.",
            "No model was trained and no performance claim follows from this report.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    try:
        if args.csv.resolve() == args.out.resolve():
            raise ValueError("Output must not overwrite the dataset.")
        report = audit(args.csv)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    print(f"Audited {report['rows']:,} rows and {len(report['columns'])} columns.")
    print(f"Report: {args.out}")
    print("Next: review the report and datasheet, then freeze the feature and split contracts.")


if __name__ == "__main__":
    main()
