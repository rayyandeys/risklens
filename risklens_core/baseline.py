"""Load audited development data and build serializable logistic pipelines."""
import csv
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import SCHEMA, FEATURES, NUMERIC, INDICATORS, CATEGORICAL, FeatureCleaner


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development(path, audit):
    fingerprint = sha256_file(path)
    if fingerprint != audit.get("sha256"):
        raise ValueError("Dataset checksum differs from the audit. Stop and investigate the file.")
    with open(path, encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source))
    if len(header) != len(SCHEMA) or set(header) != set(SCHEMA):
        raise ValueError("Dataset does not match the 32-column Base feature contract")
    pieces, counts = [], {m: 0 for m in range(8)}
    for chunk in pd.read_csv(path, chunksize=100_000):
        if not chunk["month"].isin(range(8)).all():
            raise ValueError("Invalid month values")
        for month, count in chunk["month"].value_counts().items():
            counts[int(month)] += int(count)
        # Months 6-7 are discarded before preprocessing, feature diagnostics or model fitting.
        dev = chunk.loc[chunk["month"] <= 5].copy()
        pieces.append(dev)
    if {str(k): v for k, v in counts.items()} != audit.get("month_rows"):
        raise ValueError("Month counts differ from the audit")
    if not all(counts.values()):
        raise ValueError("All eight months must be present")
    data = pd.concat(pieces)
    train = data.loc[data["month"] <= 4].copy()
    validation = data.loc[data["month"] == 5].copy()
    for name, frame in [("train", train), ("validation", validation)]:
        if not frame.fraud_bool.isin([0, 1]).all() or frame.fraud_bool.nunique() != 2:
            raise ValueError(f"{name} requires two binary classes")
        expected = audit["development_target_summary"][name]
        if len(frame) != expected["rows"] or int(frame.fraud_bool.sum()) != expected["fraud_rows"]:
            raise ValueError(f"{name} differs from audited target counts")
    return train, validation, fingerprint, counts


def diagnostics(train, validation):
    cleaner = FeatureCleaner()
    tr, va = cleaner.transform(train), cleaner.transform(validation)
    # A duplicate screen, not an entity-identity check. The benchmark has no entity IDs.
    tr_hash = pd.util.hash_pandas_object(tr, index=False)
    va_hash = pd.util.hash_pandas_object(va, index=False)
    overlap = int(va_hash.isin(tr_hash).sum())
    report = {
        "train_duplicate_feature_rows": int(tr_hash.duplicated().sum()),
        "validation_duplicate_feature_rows": int(va_hash.duplicated().sum()),
        "validation_rows_matching_training_features": overlap,
        "duplicate_check": "64-bit feature hashes; suspected matches require exact review",
        "constant_training_features": [c for c in tr if tr[c].nunique(dropna=False) <= 1],
        "training_missing_after_cleaning": {c: int(tr[c].isna().sum()) for c in FEATURES},
        "validation_unseen_categories": {
            c: int((~va[c].isin(tr[c].dropna().unique()) & va[c].notna()).sum()) for c in CATEGORICAL},
        "feature_timing": "See FEATURE_CONTRACT.md: benchmark assumptions, not production provenance",
    }
    if overlap:
        raise ValueError(f"{overlap} validation feature rows match training hashes; inspect before training")
    return report


def make_pipeline(class_weight=None, seed=42):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("missing_flags", "passthrough", INDICATORS),
        ("categorical", categorical, CATEGORICAL)], sparse_threshold=1.0)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4,
                               class_weight=class_weight, random_state=seed)
    return Pipeline([("clean", FeatureCleaner()), ("preprocess", preprocess), ("model", model)])
