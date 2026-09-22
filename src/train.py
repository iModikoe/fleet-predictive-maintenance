import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data_loader import add_training_rul, load_test_targets, load_trajectory
from src.features import SENSORS, engineer_features, model_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RANDOM_STATE = 42


def scores(actual, predicted):
    return {"mae": mean_absolute_error(actual, predicted), "rmse": mean_squared_error(actual, predicted) ** 0.5, "r2": r2_score(actual, predicted)}


def priority(rul):
    return np.select([rul <= 15, rul <= 30, rul <= 60], ["Critical", "Warning", "Monitor"], default="Normal")


def main():
    MODELS.mkdir(exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    train = engineer_features(add_training_rul(load_trajectory(RAW / "train_FD001.txt")))
    features = model_features(train)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, validation_idx = next(splitter.split(train, groups=train["unit_number"]))
    development, validation = train.iloc[train_idx], train.iloc[validation_idx]
    candidates = {
        "linear_regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        model.fit(development[features], development["rul"])
        prediction = np.clip(model.predict(validation[features]), 0, 125)
        rows.append({"model": name, **scores(validation["rul"], prediction)})
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("rmse")
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    best_rmse = comparison.iloc[0]["rmse"]
    compact_candidate = comparison.loc[comparison["model"] == "hist_gradient_boosting"].iloc[0]
    winner = "hist_gradient_boosting" if compact_candidate["rmse"] <= best_rmse * 1.01 else comparison.iloc[0]["model"]
    model = fitted[winner]
    test = engineer_features(load_trajectory(RAW / "test_FD001.txt"))
    last_rows = test.groupby("unit_number", as_index=False).tail(1).sort_values("unit_number")
    actual = load_test_targets(RAW / "RUL_FD001.txt")
    predicted = np.clip(model.predict(last_rows[features]), 0, 125)
    test_result = {"selected_model": winner, **scores(actual, predicted), "test_engines": int(len(actual)), "engine_level_split": True}
    (REPORTS / "test_metrics.json").write_text(json.dumps(test_result, indent=2))
    predictions = pd.DataFrame({"unit_number": last_rows["unit_number"].to_numpy(), "actual_rul": actual, "predicted_rul": predicted})
    predictions["absolute_error"] = (predictions["actual_rul"] - predictions["predicted_rul"]).abs()
    predictions["maintenance_priority"] = priority(predictions["predicted_rul"])
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    healthy = development[development["rul"] >= 100]
    anomaly = Pipeline([("scale", StandardScaler()), ("model", IsolationForest(contamination=0.05, random_state=RANDOM_STATE))])
    anomaly.fit(healthy[SENSORS])
    joblib.dump(model, MODELS / "rul_model.joblib", compress=3)
    joblib.dump(anomaly, MODELS / "anomaly_model.joblib", compress=3)
    metadata = {"model": winner, "features": features, "sensors": SENSORS, "rul_cap": 125, "test_metrics": test_result}
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plt.figure(figsize=(6, 6)); plt.scatter(actual, predicted, alpha=0.65); plt.plot([0, 125], [0, 125], "--", color="black")
    plt.xlabel("Actual RUL"); plt.ylabel("Predicted RUL"); plt.title("Official test engines: actual vs predicted RUL"); plt.tight_layout(); plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=160); plt.close()
    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        values = pd.DataFrame({"feature": features, "importance": importance}).nlargest(15, "importance").sort_values("importance")
        values.to_csv(REPORTS / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 6)); plt.barh(values["feature"], values["importance"]); plt.tight_layout(); plt.savefig(FIGURES / "feature_importance.png", dpi=160); plt.close()
    sample_unit = test[test["unit_number"] == test["unit_number"].min()]
    sample_unit.to_csv(ROOT / "data/sample_engine.csv", index=False)
    print(json.dumps(test_result, indent=2))


if __name__ == "__main__":
    main()
