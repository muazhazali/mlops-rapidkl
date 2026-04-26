"""Rolling inference replay: walk forward through test dates, 1 day per sleep interval."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys_path_src = PROJECT_ROOT / "src"
import sys
sys.path.insert(0, str(sys_path_src))

from mlops.pipelines.training.nodes import (
    FEATURE_COLS,
    MY_HOLIDAYS,
    _add_holiday_features,
)


def _build_features_for_date(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    lag_days: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Build feature rows for all stations for target_date."""
    stations = history["station"].unique()
    rows = []

    for station in stations:
        h = history[history["station"] == station].sort_values("date")
        past = h[h["date"] < target_date]
        if past.empty:
            continue

        row: dict = {
            "station": station,
            "date": target_date,
            "year": target_date.year,
            "month": target_date.month,
            "day": target_date.day,
            "day_of_week": target_date.dayofweek,
            "is_weekend": int(target_date.dayofweek >= 5),
        }

        for lag in lag_days:
            lag_date = target_date - pd.Timedelta(days=lag)
            match = past[past["date"] == lag_date]
            row[f"lag_{lag}"] = match["ridership"].values[0] if not match.empty else np.nan

        for w in rolling_windows:
            row[f"rolling_mean_{w}"] = past["ridership"].tail(w).mean()

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    feat = pd.DataFrame(rows)
    feat["date"] = pd.to_datetime(feat["date"])
    feat = _add_holiday_features(feat)
    feat = feat.dropna(subset=[f"lag_{lag_days[-1]}"])
    return feat


def run_inference() -> None:
    with open(PROJECT_ROOT / "conf" / "base" / "parameters.yml") as f:
        params = yaml.safe_load(f)

    model_path = PROJECT_ROOT / "data" / "06_models" / "catboost_model.pkl"
    station_daily_path = PROJECT_ROOT / "data" / "02_intermediate" / "station_daily.parquet"
    actuals_path = PROJECT_ROOT / "data" / "07_model_output" / "actuals.parquet"
    predictions_path = PROJECT_ROOT / "data" / "07_model_output" / "predictions.parquet"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    history = pd.read_parquet(station_daily_path)
    history["date"] = pd.to_datetime(history["date"])

    runner_cfg = params["pipeline_runner"]
    inference_start = pd.to_datetime(runner_cfg["inference_start"])
    interval_sec = runner_cfg["inference_interval_seconds"]

    lag_days: list[int] = params["feature_engineering"]["lag_days"]
    rolling_windows: list[int] = params["feature_engineering"]["rolling_windows"]

    test_dates = sorted(history[history["date"] >= inference_start]["date"].unique())

    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        if predictions_path.exists():
            predictions_path.unlink()
            print("Cleared previous predictions.")

        print(f"Starting inference replay: {len(test_dates)} dates")

        for i, target_date in enumerate(test_dates):
            feat = _build_features_for_date(history, target_date, lag_days, rolling_windows)
            if feat.empty:
                continue

            preds_raw = model.predict(feat[FEATURE_COLS])
            preds_raw = np.maximum(preds_raw, 0)

            result = feat[["station", "date"]].copy()
            result["prediction"] = preds_raw.round().astype(int)

            if predictions_path.exists():
                existing = pd.read_parquet(predictions_path)
                result = pd.concat([existing, result], ignore_index=True)
            result.to_parquet(predictions_path, index=False)

            print(f"[{i + 1}/{len(test_dates)}] {target_date.date()} — {len(feat)} stations predicted")

            if i < len(test_dates) - 1:
                time.sleep(interval_sec)

        print("Replay complete. Restarting...")


if __name__ == "__main__":
    run_inference()
