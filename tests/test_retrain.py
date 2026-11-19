import pandas as pd

from src import retrain


def test_check_drift_detects_drift(tmp_path, monkeypatch):
    drift_file = tmp_path / "drift_results.csv"

    pd.DataFrame({
        "feature": ["MedInc", "HouseAge", "AveRooms", "Population"],
        "drift_detected": [True, False, False, False]
    }).to_csv(drift_file, index=False)

    monkeypatch.setattr(retrain, "DRIFT_RESULTS_PATH", drift_file)

    drift_detected, drift_share = retrain.check_drift()

    assert drift_detected is True
    assert drift_share == 0.25


def test_check_drift_does_not_detect_drift(tmp_path, monkeypatch):
    drift_file = tmp_path / "drift_results.csv"

    pd.DataFrame({
        "feature": ["MedInc", "HouseAge", "AveRooms", "Population"],
        "drift_detected": [False, False, False, False]
    }).to_csv(drift_file, index=False)

    monkeypatch.setattr(retrain, "DRIFT_RESULTS_PATH", drift_file)

    drift_detected, drift_share = retrain.check_drift()

    assert drift_detected is False
    assert drift_share == 0.0


def test_evaluate_model():
    class DummyModel:
        def predict(self, X):
            return [2.0, 3.0, 4.0]

    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    y_test = pd.Series([2.0, 4.0, 4.0])

    metrics = retrain.evaluate_model(DummyModel(), X_test, y_test)

    assert metrics["MAE"] >= 0
    assert metrics["MSE"] >= 0
    assert metrics["RMSE"] >= 0
    assert -1 <= metrics["R2"] <= 1
