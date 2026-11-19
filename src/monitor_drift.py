from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from evidently import Report
from evidently.presets import DataDriftPreset


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = REPORT_DIR / "data_drift_report.html"
CURRENT_DATA_PATH = REPORT_DIR / "current_housing_data.csv"
DRIFT_RESULTS_PATH = REPORT_DIR / "drift_results.csv"

CURRENT_DATA_SIZE = 2000
P_VALUE_THRESHOLD = 0.05
KS_STATISTIC_THRESHOLD = 0.10
DRIFT_SHARE_THRESHOLD = 0.25

# Target is excluded because we monitor input feature drift.
TARGET_COLUMN = "MedHouseVal"


# ==========================================
# LOAD REFERENCE DATA
# ==========================================

def load_reference_data():

    print("\nLoading reference dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column {TARGET_COLUMN} is missing."
        )

    print("Reference dataset shape:", data.shape)

    return data


# ==========================================
# GENERATE SIMULATED CURRENT DATA
# ==========================================

def generate_current_data(reference_data):

    print("\nGenerating simulated incoming data...")

    current_data = reference_data.sample(
        n=CURRENT_DATA_SIZE,
        random_state=42
    ).copy()

    # Artificially introduce feature distribution changes.
    # This is simulated data for testing drift detection.

    current_data["MedInc"] *= 1.5

    current_data["HouseAge"] = (
        current_data["HouseAge"] + 10
    ).clip(upper=52)

    current_data["AveOccup"] *= 1.5

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    current_data.to_csv(
        CURRENT_DATA_PATH,
        index=False
    )

    print("Current dataset shape:", current_data.shape)
    print("Saved current data:", CURRENT_DATA_PATH)

    return current_data


# ==========================================
# EVIDENTLY HTML REPORT
# ==========================================

def generate_evidently_report(reference_data, current_data):

    print("\nGenerating Evidently AI report...")

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    snapshot = report.run(
        current_data=current_data,
        reference_data=reference_data
    )

    snapshot.save_html(str(REPORT_PATH))

    print("Evidently report saved:", REPORT_PATH)


# ==========================================
# STATISTICAL DRIFT DETECTION
# ==========================================

def detect_drift(reference_data, current_data):

    print("\nRunning statistical drift detection...")

    # Exclude target column from feature monitoring.
    reference_features = reference_data.drop(
        columns=[TARGET_COLUMN]
    )

    current_features = current_data.drop(
        columns=[TARGET_COLUMN]
    )

    results = []

    for column in reference_features.columns:

        # Kolmogorov-Smirnov test compares two distributions.
        statistic, p_value = ks_2samp(
            reference_features[column],
            current_features[column]
        )

        column_drift = (
            p_value < P_VALUE_THRESHOLD
            and statistic >= KS_STATISTIC_THRESHOLD
        )

        results.append({
            "feature": column,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": column_drift
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        DRIFT_RESULTS_PATH,
        index=False
    )

    print("\n========== FEATURE DRIFT RESULTS ==========")

    print(results_df.to_string(index=False))

    drifted_count = int(
        results_df["drift_detected"].sum()
    )

    total_features = len(results_df)

    drift_share = drifted_count / total_features

    drift_detected = (
        drift_share >= DRIFT_SHARE_THRESHOLD
    )

    print("\n========== DRIFT SUMMARY ==========")

    print("Drifted features:", drifted_count)
    print("Total features:", total_features)
    print(f"Drift share: {drift_share:.2%}")

    print(
        f"Required drift share: "
        f"{DRIFT_SHARE_THRESHOLD:.2%}"
    )

    print("Dataset drift detected:", drift_detected)

    return drift_detected


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    print("\n========================================")
    print("       HOUSING DATA DRIFT MONITOR")
    print("========================================")

    # Step 1: Load original data
    reference_data = load_reference_data()

    # Step 2: Generate simulated incoming data
    current_data = generate_current_data(
        reference_data
    )

    # Step 3: Generate Evidently HTML report
    generate_evidently_report(
        reference_data,
        current_data
    )

    # Step 4: Detect feature drift
    drift_detected = detect_drift(
        reference_data,
        current_data
    )

    # Step 5: Make pipeline decision
    print("\n========== PIPELINE DECISION ==========")

    if drift_detected:

        print("STATUS: DATA DRIFT DETECTED")

        print("ACTION: RETRAINING RECOMMENDED")

        print(
            "Collect correctly labelled new data "
            "before training a replacement model."
        )

    else:

        print("STATUS: NO SIGNIFICANT DATA DRIFT")

        print("ACTION: KEEP EXISTING MODEL")

    print("\n========== GENERATED REPORTS ==========")

    print("HTML report:", REPORT_PATH)
    print("Current data:", CURRENT_DATA_PATH)
    print("Drift results:", DRIFT_RESULTS_PATH)

    print("\nMonitoring completed successfully.")


if __name__ == "__main__":
    main()
