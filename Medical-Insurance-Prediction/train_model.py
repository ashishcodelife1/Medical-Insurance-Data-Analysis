"""Train the Linear Regression pipeline and save the model + real metrics.

Run from the project root:   python train_model.py
"""
import argparse
import sys
from pathlib import Path

from src.model_utils import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    DatasetError,
    load_dataset,
    save_artifacts,
    train_and_evaluate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the insurance cost model.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to insurance.csv")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data}")
    try:
        df = load_dataset(Path(args.data))
    except DatasetError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    print(f"Dataset OK: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Training Linear Regression (80% train / 20% test)...")
    pipeline, metrics = train_and_evaluate(df)
    save_artifacts(pipeline, metrics)

    print("\n=== Evaluation on the held-out test set ===")
    print(f"MAE  : {metrics['mae']:,.2f} USD")
    print(f"RMSE : {metrics['rmse']:,.2f} USD")
    print(f"R2   : {metrics['r2']:.4f}")
    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print("\nNext: python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
