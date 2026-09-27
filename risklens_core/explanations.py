"""Reproducible sampled permutation contributions on the raw model inputs.

Each training background row uses a random order and its reverse. Whole raw
features are replaced before the saved cleaner runs, so missing flags stay paired
with their source feature. This estimates empirical interventional Shapley values;
it is not exact TreeSHAP, causal inference, or a probability calibration method.
"""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pandas as pd

from .baseline import sha256_file
from .features import CONTRACT_VERSION, FEATURES, SCHEMA, FeatureCleaner

METHOD = "paired-permutation-v1"
LIMITATIONS = [
    "Approximate model attributions, not causes of fraud or proof of wrongdoing.",
    "Replacing features can create unrealistic combinations when inputs are correlated.",
    "Reference rows are a small sample from training months 0-4; results depend on that sample.",
    "pair_std_dev describes variation across background/order pairs; it is not a confidence interval.",
    "The model score is uncalibrated; contributions are score differences, not established fraud probabilities.",
    "Explanations do not establish fairness; age/proxy and subgroup analysis remain separate work.",
]


def json_value(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def feature_hash(frame):
    validate_features(frame)
    return canonical_hash({"features": FEATURES, "rows": [
        [json_value(v) for v in row] for row in frame[FEATURES].itertuples(index=False, name=None)
    ]})


def validate_features(frame):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any() or set(frame.columns) != set(FEATURES):
        raise ValueError("Explanation input must contain exactly the label-free raw feature contract")
    if frame.empty:
        raise ValueError("Explanation input is empty")
    FeatureCleaner().transform(frame)


def load_training_background(path, audit, *, size=16, seed=42):
    """Read feature/month columns only. Never load targets for explanations."""
    fingerprint = sha256_file(path)
    if fingerprint != audit.get("sha256"):
        raise ValueError("Dataset checksum differs from audit")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if len(header) != len(SCHEMA) or set(header) != set(SCHEMA):
        raise ValueError("Unexpected Base.csv schema")
    count = int(audit["development_target_summary"]["train"]["rows"])
    if not isinstance(size, int) or not 2 <= size <= min(count, 128):
        raise ValueError("Background size must be between 2 and 128 and fit the training population")
    positions = np.sort(np.random.default_rng(seed).choice(count, size=size, replace=False))
    offset = 0
    selected = []
    counts = {m: 0 for m in range(8)}
    for chunk in pd.read_csv(path, usecols=[*FEATURES, "month"], chunksize=100_000):
        if not chunk.month.isin(range(8)).all():
            raise ValueError("Invalid month")
        for month, rows in chunk.month.value_counts().items():
            counts[int(month)] += int(rows)
        training = chunk.loc[chunk.month <= 4, FEATURES]
        local = positions[(positions >= offset) & (positions < offset + len(training))] - offset
        if len(local):
            selected.append(training.iloc[local])
        offset += len(training)
    if offset != count or {str(k): v for k, v in counts.items()} != audit.get("month_rows"):
        raise ValueError("Audited month/training counts changed")
    background = pd.concat(selected)
    validate_features(background)
    metadata = {"dataset_sha256": fingerprint, "months": [0, 1, 2, 3, 4],
                "row_ids": [int(i) for i in background.index], "size": size, "seed": seed,
                "features_sha256": feature_hash(background)}
    metadata["background_sha256"] = canonical_hash(metadata)
    return background, metadata


def explain_case(model, case, background, *, seed=42):
    validate_features(case)
    validate_features(background)
    if len(case) != 1 or not 2 <= len(background) <= 128:
        raise ValueError("Explain one case with 2-128 reference rows")
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Expected binary model classes [0, 1]")
    rng = np.random.default_rng(seed)
    x = case[FEATURES].iloc[0].tolist()
    orders, paths = [], []
    width = len(FEATURES)
    for row in background[FEATURES].itertuples(index=False, name=None):
        order = rng.permutation(width)
        for permutation in (order, order[::-1]):
            current = list(row)
            paths.append(current.copy())
            for index in permutation:
                current[index] = x[index]
                paths.append(current.copy())
            orders.append(permutation)
    hybrid = pd.DataFrame(paths, columns=FEATURES)
    probabilities = np.asarray(model.predict_proba(hybrid), dtype=float)
    if probabilities.shape != (len(hybrid), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Model returned invalid probabilities")
    scores = probabilities[:, 1].reshape(len(orders), width + 1)
    if not np.allclose(scores[:, -1], scores[0, -1], rtol=0, atol=1e-12):
        raise ValueError("Model predictions are not deterministic for the same case")
    contributions = np.zeros((len(orders), width))
    for index, order in enumerate(orders):
        contributions[index, order] = np.diff(scores[index])
    paired = contributions.reshape(len(background), 2, width).mean(axis=1)
    mean = paired.mean(axis=0)
    std = paired.std(axis=0, ddof=1)
    base = float(scores[:, 0].mean())
    score = float(scores[0, -1])
    residual = abs(base + mean.sum() - score)
    if residual > 1e-10:
        raise ValueError("Explanation fails additive reconstruction")
    cleaned = FeatureCleaner().transform(case).iloc[0]
    features = [{"feature": name, "value": json_value(x[i]),
                 "missing_after_cleaning": bool(pd.isna(cleaned[name])),
                 "contribution": float(mean[i]), "pair_std_dev": float(std[i])}
                for i, name in enumerate(FEATURES)]
    features.sort(key=lambda item: (-abs(item["contribution"]), item["feature"]))
    return {"method": METHOD, "feature_contract": CONTRACT_VERSION,
            "score_scale": "uncalibrated_model_score", "reference_score": base,
            "model_score": score, "reconstruction_error": float(residual),
            "background_size": len(background), "permutation_paths": len(orders),
            "model_rows_evaluated": len(hybrid), "seed": int(seed),
            "features": features, "limitations": LIMITATIONS.copy()}
