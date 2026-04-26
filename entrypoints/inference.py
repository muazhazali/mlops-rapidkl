"""Rolling inference: replay known dates then autoregressive future forecast."""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlops.pipelines.training.nodes import (
    FEATURE_COLS,
    _add_holiday_features,
)


def _build_features_for_date(
    working_history: pd.DataFrame,
    target_date: pd.Timestamp,
    lag_days: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Build feature rows for all stations for target_date using working_history.

    working_history contains both actuals and previously predicted values,
    enabling autoregressive forecasting beyond the dataset end.
    """
    stations = working_history["station"].unique()
    rows = []

    for station in stations:
        h = working_history[working_history["station"] == station].sort_values("date")
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


def _predict_and_append(
    model,
    feat: pd.DataFrame,
    predictions_path: Path,
) -> pd.DataFrame:
    preds_raw = np.maximum(model.predict(feat[FEATURE_COLS]), 0)
    result = feat[["station", "date"]].copy()
    result["prediction"] = preds_raw.round().astype(int)

    if predictions_path.exists():
        existing = pd.read_parquet(predictions_path)
        result = pd.concat([existing, result], ignore_index=True)
    result.to_parquet(predictions_path, index=False)
    return result


def run_inference() -> None:
    with open(PROJECT_ROOT / "conf" / "base" / "parameters.yml") as f:
        params = yaml.safe_load(f)

    model_path = PROJECT_ROOT / "data" / "06_models" / "catboost_model.pkl"
    station_daily_path = PROJECT_ROOT / "data" / "02_intermediate" / "station_daily.parquet"
    predictions_path = PROJECT_ROOT / "data" / "07_model_output" / "predictions.parquet"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    actuals = pd.read_parquet(station_daily_path)
    actuals["date"] = pd.to_datetime(actuals["date"])

    runner_cfg = params["pipeline_runner"]
    inference_start = pd.to_datetime(runner_cfg["inference_start"])
    interval_sec = runner_cfg["inference_interval_seconds"]
    future_days: int = runner_cfg.get("future_forecast_days", 30)

    lag_days: list[int] = params["feature_engineering"]["lag_days"]
    rolling_windows: list[int] = params["feature_engineering"]["rolling_windows"]

    # Known test dates from dataset + N future dates beyond last known date
    known_dates = sorted(actuals[actuals["date"] >= inference_start]["date"].unique())
    last_known = actuals["date"].max()
    future_dates = [
        last_known + pd.Timedelta(days=i) for i in range(1, future_days + 1)
    ]
    all_dates = known_dates + future_dates

    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        if predictions_path.exists():
            predictions_path.unlink()
            print("Cleared previous predictions.")

        # working_history = actuals; future predictions are appended here so
        # subsequent lag features reference them instead of missing data
        working_history = actuals.copy()

        print(f"Replay: {len(known_dates)} known + {future_days} future dates")

        for i, target_date in enumerate(all_dates):
            is_future = target_date > last_known
            label = f"[FUTURE +{(target_date - last_known).days}d]" if is_future else f"[{i + 1}/{len(all_dates)}]"

            feat = _build_features_for_date(working_history, target_date, lag_days, rolling_windows)
            if feat.empty:
                print(f"{label} {target_date.date()} — skipped (insufficient history)")
                continue

            _predict_and_append(model, feat, predictions_path)

            if is_future:
                # Append predictions as synthetic actuals so next lags are correct
                synthetic = feat[["station", "date"]].copy()
                preds_raw = np.maximum(model.predict(feat[FEATURE_COLS]), 0)
                synthetic["ridership"] = preds_raw.round().astype(int)
                working_history = pd.concat([working_history, synthetic], ignore_index=True)

            print(f"{label} {target_date.date()} — {len(feat)} stations")

            if i < len(all_dates) - 1:
                time.sleep(interval_sec)

        print("Cycle complete. Restarting...")


if __name__ == "__main__":
    run_inference()
