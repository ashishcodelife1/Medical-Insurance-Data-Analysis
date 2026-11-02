"""Shared helpers used by BOTH train_model.py and app.py.

Keeping everything here guarantees that training and prediction use the same
feature names, categories and preprocessing pipeline.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "insurance.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "insurance_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ------------------------------------------------------- feature schema ---
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "charges"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]

CATEGORY_VALUES = {
    "sex": ["female", "male"],
    "smoker": ["no", "yes"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

DATASET_HELP = (
    "Download insurance.csv (Medical Cost Personal Dataset) and save it as "
    "'data/insurance.csv' inside the project folder. Required columns: "
    "age, sex, bmi, children, smoker, region, charges."
)


class DatasetError(Exception):
    """Raised when the dataset is missing or invalid."""


class ModelNotFoundError(Exception):
    """Raised when the trained model or metrics file is missing."""


# --------------------------------------------------------------- dataset ---
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Check the dataset and return a cleaned copy. Raises DatasetError."""
    if df.empty:
        raise DatasetError("The dataset file is empty.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            f"The dataset is missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )
    df = df[REQUIRED_COLUMNS].copy()

    null_counts = df.isna().sum()
    if null_counts.sum() > 0:
        bad = null_counts[null_counts > 0].to_dict()
        raise DatasetError(f"The dataset contains missing values: {bad}.")

    for col in NUMERIC_FEATURES + [TARGET]:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise DatasetError(f"Column '{col}' contains non-numeric values.")
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise DatasetError(f"Column '{col}' contains infinite values.")
        df[col] = converted

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
        unexpected = sorted(set(df[col]) - set(CATEGORY_VALUES[col]))
        if unexpected:
            raise DatasetError(
                f"Column '{col}' has unexpected value(s): {unexpected}. "
                f"Allowed: {CATEGORY_VALUES[col]}."
            )

    if (df["age"] <= 0).any():
        raise DatasetError("Column 'age' must be positive.")
    if (df["bmi"] <= 0).any():
        raise DatasetError("Column 'bmi' must be positive.")
    if (df["children"] < 0).any():
        raise DatasetError("Column 'children' cannot be negative.")
    if (df[TARGET] <= 0).any():
        raise DatasetError("Column 'charges' must be positive.")

    df["children"] = df["children"].astype(int)
    return df.reset_index(drop=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the CSV file."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found at '{path}'. {DATASET_HELP}")
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # unreadable / corrupted CSV
        raise DatasetError(f"The CSV file could not be read: {exc}") from exc
    return validate_dataframe(raw)


# -------------------------------------------------------------- training ---
def build_pipeline() -> Pipeline:
    """Preprocessing (one-hot encoding) + Linear Regression in one pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORY_VALUES[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def train_and_evaluate(df: pd.DataFrame):
    """Split, fit on the training set only, evaluate on the test set."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)  # encoder is fitted on training data only
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model_type": "Linear Regression",
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "test_predictions": {
            "actual": [float(v) for v in y_test],
            "predicted": [float(v) for v in y_pred],
        },
    }
    return pipeline, metrics


def save_artifacts(pipeline: Pipeline, metrics: dict) -> None:
    """Save the fitted pipeline and the real evaluation results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


# ------------------------------------------------------------ prediction ---
def load_model(path: Path = MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            "Trained model not found. Run 'python train_model.py' in the terminal first."
        )
    return joblib.load(path)


def load_metrics(path: Path = METRICS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            "Metrics file not found. Run 'python train_model.py' in the terminal first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def predict_charge(model: Pipeline, inputs: dict) -> float:
    """Predict charges (USD) for one person using the saved pipeline."""
    row = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    value = float(model.predict(row)[0])
    if not np.isfinite(value):
        raise ValueError("The model returned a non-finite prediction.")
    return value
