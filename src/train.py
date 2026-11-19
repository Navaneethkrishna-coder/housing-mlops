
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost_housing_model.pkl"

# MLflow configuration
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "Housing-Price-Prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def train_model():

    print("Loading housing dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run python src/data_loader.py first."
        )

    df = pd.read_csv(DATA_PATH)

    # Separate input features and target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Model configuration
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "reg:squarederror",
        "n_jobs": -1
    }

    model = XGBRegressor(**params)

    # MLflow experiment
    with mlflow.start_run(run_name="xgboost-housing-v1"):

        print("\nTraining XGBoost model...")

        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        print("\nModel Evaluation Results")
        print("------------------------")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # Log parameters and metrics
        mlflow.log_params(params)

        mlflow.log_metrics({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        # Log model to MLflow
        mlflow.xgboost.log_model(
            model,
            name="xgboost_housing_model"
        )

        # Save model locally
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        print(f"\nModel saved successfully: {MODEL_PATH}")

        print("MLflow experiment completed.")

    return model


if __name__ == "__main__":
    train_model()

