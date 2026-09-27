"""Fixed PyTorch tabular baseline for the RiskLens development comparison.

The module intentionally keeps the neural experiment small and auditable. It uses the
canonical RiskLens cleaning contract, train-fitted dense preprocessing, a fixed MLP,
and a fixed training schedule. Month 5 is never used for early stopping or optimizer
selection; months 6-7 are never loaded by the experiment CLI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import random
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, INDICATORS, NUMERIC, FeatureCleaner

try:  # Keep import failure readable when the new dependency has not been installed yet.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only in an environment without torch
    torch = None
    nn = None
    DataLoader = TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class TorchMLPConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    epochs: int = 15
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    class_weighting: str = "none"

    def to_json(self) -> dict:
        value = asdict(self)
        value["hidden_dims"] = list(self.hidden_dims)
        return value


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for torch-comparison-v1. Run `python -m pip install -r requirements.txt`."
        ) from _TORCH_IMPORT_ERROR


def make_preprocessor() -> Pipeline:
    """Build the train-fitted dense preprocessing used by the neural baseline.

    Numerical values are median-imputed and standardized. Canonical missing flags pass
    through unchanged. Categorical values are most-frequent-imputed and one-hot encoded,
    with unseen validation categories mapped to all-zero indicator blocks.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("missing_flags", "passthrough", INDICATORS),
        ("categorical", categorical, CATEGORICAL),
    ], sparse_threshold=0)
    return Pipeline([
        ("clean", FeatureCleaner()),
        ("transform", transform),
    ])


def fit_transform_preprocessor(train: pd.DataFrame, validation: pd.DataFrame):
    preprocessor = make_preprocessor()
    train_matrix = preprocessor.fit_transform(train[FEATURES]).astype(np.float32, copy=False)
    validation_matrix = preprocessor.transform(validation[FEATURES]).astype(np.float32, copy=False)
    if train_matrix.ndim != 2 or validation_matrix.ndim != 2:
        raise ValueError("Neural preprocessing must produce 2D matrices")
    if train_matrix.shape[1] != validation_matrix.shape[1] or train_matrix.shape[1] == 0:
        raise ValueError("Train and validation neural feature dimensions differ")
    if not np.isfinite(train_matrix).all() or not np.isfinite(validation_matrix).all():
        raise ValueError("Neural preprocessing produced non-finite values")
    return preprocessor, train_matrix, validation_matrix


if nn is not None:
    class TabularMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (128, 64), dropout: float = 0.10):
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be positive")
            hidden_dims = tuple(int(v) for v in hidden_dims)
            if not hidden_dims or any(v <= 0 for v in hidden_dims):
                raise ValueError("hidden_dims must contain positive widths")
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in [0, 1)")
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend([
                    nn.Linear(previous, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features):
            return self.network(features).squeeze(-1)
else:  # pragma: no cover
    class TabularMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()


def seed_everything(seed: int, *, threads: int = 4) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is the default experiment device. Thread bound prevents accidental laptop oversubscription.
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting interop threads only before parallel work begins in a process.
        pass


def resolve_device(requested: str = "cpu") -> str:
    require_torch()
    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def _positive_weight(y: np.ndarray, mode: str):
    if mode == "none":
        return None
    if mode != "inverse_frequency":
        raise ValueError("Unknown class weighting mode")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels require both classes")
    return float(negatives / positives)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: TorchMLPConfig = TorchMLPConfig(),
    device: str = "cpu",
    threads: int = 4,
):
    """Fit the fixed MLP for a fixed number of epochs and return model + history.

    There is intentionally no month-5 early stopping. The validation month remains an
    evaluation set rather than an optimizer feedback channel.
    """
    require_torch()
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if x_train.ndim != 2 or y_train.ndim != 1 or len(x_train) != len(y_train):
        raise ValueError("Training matrix and labels are misaligned")
    if len(x_train) == 0 or not np.isin(y_train, [0.0, 1.0]).all() or len(np.unique(y_train)) != 2:
        raise ValueError("Training labels must be nonempty and binary with two classes")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    seed_everything(config.seed, threads=threads)
    device = resolve_device(device)
    model = TabularMLP(x_train.shape[1], config.hidden_dims, config.dropout).to(device)
    positive_weight = _positive_weight(y_train.astype(int), config.class_weighting)
    pos_tensor = None if positive_weight is None else torch.tensor(positive_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite neural training loss")
            loss.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch = len(labels)
            total_loss += float(loss.detach().cpu()) * batch
            examples += batch
        history.append({"epoch": epoch, "mean_training_loss": total_loss / examples})
    model.eval()
    return model, history, {"positive_class_weight": positive_weight, "device": device}


def predict_scores(model, matrix: np.ndarray, *, device: str = "cpu", batch_size: int = 16384) -> np.ndarray:
    require_torch()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("Scoring matrix must be a nonempty 2D array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            features = torch.from_numpy(matrix[start:start + batch_size]).to(device)
            logits = model(features)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            chunks.append(probabilities)
    result = np.concatenate(chunks).astype(float, copy=False)
    if result.shape != (len(matrix),) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Neural scorer produced invalid scores")
    return result


def save_torch_bundle(path, model, *, input_dim: int, config: TorchMLPConfig, metadata: dict | None = None) -> None:
    require_torch()
    payload = {
        "format": "risklens-torch-mlp-v1",
        "input_dim": int(input_dim),
        "config": config.to_json(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_torch_bundle(path, *, device: str = "cpu"):
    require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "risklens-torch-mlp-v1":
        raise ValueError("Unrecognized RiskLens torch bundle format")
    cfg = payload.get("config", {})
    config = TorchMLPConfig(
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        gradient_clip_norm=float(cfg["gradient_clip_norm"]),
        seed=int(cfg["seed"]),
        class_weighting=str(cfg["class_weighting"]),
    )
    model = TabularMLP(int(payload["input_dim"]), config.hidden_dims, config.dropout)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(device))
    model.eval()
    return model, config, payload.get("metadata", {})


def parameter_count(model) -> int:
    require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters()))
