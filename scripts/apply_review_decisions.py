"""Apply a completed analyst decision CSV to a RiskLens queue snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from risklens_core.review import apply_review_decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    queue = pd.read_csv(args.queue, keep_default_na=False)
    decisions = pd.read_csv(args.decisions, keep_default_na=False)
    decisions = decisions.loc[decisions["decision"].astype(str).str.len() > 0].copy()
    updated = apply_review_decisions(queue, decisions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.out, index=False)
    print(f"Applied {len(decisions)} analyst decisions -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Decision update stopped: {exc}") from exc
