
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "housing.csv"


def load_housing_data():
    """Download and save the California Housing dataset."""

    print("Loading California Housing dataset...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()

    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    return df


def validate_data(df):
    """Validate the downloaded dataset."""

    print("\n--- Dataset Validation ---")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe())

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if "MedHouseVal" not in df.columns:
        raise ValueError("Target column MedHouseVal is missing.")

    print("\nDataset validation successful!")


if __name__ == "__main__":
    housing_data = load_housing_data()
    validate_data(housing_data)

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_housing_data():
    """Download and load the California Housing dataset."""

    print("Loading California Housing dataset...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)

    df = dataset.frame.copy()

    output_path = DATA_DIR / "housing.csv"

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    data = load_housing_data()

    print("\nFirst five records:")
    print(data.head())

    print("\nDataset information:")
    print(data.info())

    print("\nMissing values:")
    print(data.isnull().sum())

