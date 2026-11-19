
from pathlib import Path
import shutil

import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"

DRIFT_RESULTS_PATH = PROJECT_ROOT / "reports" / "drift_results.csv"

MODEL_DIR = PROJECT_ROOT / "models"

CURRENT_MODEL_PATH = MODEL_DIR / "xgboost_housing_model.pkl"

CANDIDATE_MODEL_PATH = MODEL_DIR / "xgboost_housing_candidate.pkl"

BACKUP_MODEL_PATH = MODEL_DIR / "xgboost_housing_previous.pkl"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

EXPERIMENT_NAME = "Housing-Price-Retraining"

DRIFT_SHARE_THRESHOLD = 0.25

# Candidate must improve MAE by at least 1%.
MIN_MAE_IMPROVEMENT = 0.01


# ==========================================
# XGBOOST CONFIGURATION
# ==========================================

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
    "n_jobs": -1
}


# ==========================================
# LOAD AND CHECK DRIFT RESULTS
# ==========================================

def check_drift():

    print("\nChecking drift results...")

    if not DRIFT_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Drift results not found. "
            "Run python src/monitor_drift.py first."
        )

    drift_data = pd.read_csv(DRIFT_RESULTS_PATH)

    required_columns = {
        "feature",
        "drift_detected"
    }

    if not required_columns.issubset(drift_data.columns):
        raise ValueError(
            "Drift results CSV has an invalid format."
        )

    drift_flags = (
        drift_data["drift_detected"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if drift_flags.isna().any():
        raise ValueError(
            "Invalid drift_detected values in CSV."
        )

    drifted_count = int(drift_flags.sum())

    total_features = len(drift_flags)

    if total_features == 0:
        raise ValueError("No features found in drift results.")

    drift_share = drifted_count / total_features

    drift_detected = (
        drift_share >= DRIFT_SHARE_THRESHOLD
    )

    print("Drifted features:", drifted_count)
    print("Total features:", total_features)
    print(f"Drift share: {drift_share:.2%}")
    print("Drift detected:", drift_detected)

    return drift_detected, drift_share


# ==========================================
# LOAD LABELLED TRAINING DATA
# ==========================================

def load_training_data():

    print("\nLoading labelled training data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ==========================================
# CONDITIONAL RETRAINING
# ==========================================

def retrain_model():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not CURRENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Existing model not found. "
            "Run python src/train.py first."
        )

    # Load the existing model
    current_model = joblib.load(CURRENT_MODEL_PATH)

    # Load labelled data
    X_train, X_test, y_train, y_test = load_training_data()

    # Evaluate existing model
    baseline_metrics = evaluate_model(
        current_model,
        X_test,
        y_test
    )

    print("\nExisting Model Metrics:")
    print(baseline_metrics)

    # Train candidate model
    print("\nTraining candidate XGBoost model...")

    candidate_model = XGBRegressor(**MODEL_PARAMS)

    candidate_model.fit(X_train, y_train)

    # Evaluate candidate
    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test
    )

    print("\nCandidate Model Metrics:")
    print(candidate_metrics)

    # Save candidate separately
    joblib.dump(
        candidate_model,
        CANDIDATE_MODEL_PATH
    )

    # Promotion criteria
    required_mae = (
        baseline_metrics["MAE"]
        * (1 - MIN_MAE_IMPROVEMENT)
    )

    candidate_approved = (
        candidate_metrics["MAE"] < required_mae
        and candidate_metrics["RMSE"] <= baseline_metrics["RMSE"]
    )

    # Record experiment in MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="conditional-xgboost-retraining"
    ):

        mlflow.log_params(MODEL_PARAMS)

        mlflow.log_metrics({
            "baseline_MAE": baseline_metrics["MAE"],
            "baseline_RMSE": baseline_metrics["RMSE"],
            "candidate_MAE": candidate_metrics["MAE"],
            "candidate_RMSE": candidate_metrics["RMSE"],
            "candidate_R2": candidate_metrics["R2"]
        })

        mlflow.log_artifact(
            str(CANDIDATE_MODEL_PATH)
        )

        mlflow.set_tag(
            "candidate_approved",
            str(candidate_approved)
        )

        if candidate_approved:

            # Preserve the previous model
            shutil.copy2(
                CURRENT_MODEL_PATH,
                BACKUP_MODEL_PATH
            )

            # Promote the candidate
            shutil.copy2(
                CANDIDATE_MODEL_PATH,
                CURRENT_MODEL_PATH
            )

            print("\nCANDIDATE APPROVED")

            print("Previous model backed up.")

            print("New model promoted successfully.")

        else:

            print("\nCANDIDATE NOT APPROVED")

            print(
                "Existing model will remain unchanged."
            )

            print(
                "Candidate must improve MAE by at least "
                f"{MIN_MAE_IMPROVEMENT:.0%} "
                "without increasing RMSE."
            )

    return candidate_approved


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    print("\n========================================")
    print("   CONDITIONAL MODEL RETRAINING")
    print("========================================")

    drift_detected, drift_share = check_drift()

    if not drift_detected:

        print("\nNo significant drift detected.")

        print("Skipping model retraining.")

        print("Existing model remains unchanged.")

        return

    print("\nDrift detected. Starting retraining workflow.")

    print(
        "\nNOTE: This demonstration retrains using "
        "the original labelled dataset."
    )

    print(
        "It does not use artificially modified "
        "current data as training data."
    )

    approved = retrain_model()

    print("\n========== FINAL DECISION ==========")

    if approved:
        print("STATUS: NEW MODEL PROMOTED")
    else:
        print("STATUS: EXISTING MODEL RETAINED")

    print("\nConditional retraining completed.")


if __name__ == "__main__":
    main()

