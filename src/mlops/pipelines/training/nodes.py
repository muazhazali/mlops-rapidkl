from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MY_HOLIDAYS: dict[str, str] = {
    "2023-01-01": "New Year's Day",
    "2023-01-22": "Chinese New Year",
    "2023-01-23": "Chinese New Year Holiday",
    "2023-02-01": "Federal Territory Day",
    "2023-04-22": "Hari Raya Aidilfitri",
    "2023-04-23": "Hari Raya Aidilfitri Holiday",
    "2023-05-01": "Labour Day",
    "2023-05-04": "Wesak Day",
    "2023-06-05": "Agong's Birthday",
    "2023-06-29": "Hari Raya Haji",
    "2023-07-19": "Awal Muharram",
    "2023-08-31": "National Day",
    "2023-09-16": "Malaysia Day",
    "2023-09-28": "Maulidur Rasul",
    "2023-11-13": "Deepavali",
    "2023-12-25": "Christmas Day",
    "2024-01-01": "New Year's Day",
    "2024-01-25": "Thaipusam",
    "2024-02-01": "Federal Territory Day",
    "2024-02-10": "Chinese New Year",
    "2024-02-11": "Chinese New Year Holiday",
    "2024-04-10": "Hari Raya Aidilfitri",
    "2024-04-11": "Hari Raya Aidilfitri Holiday",
    "2024-05-01": "Labour Day",
    "2024-05-22": "Wesak Day",
    "2024-06-03": "Agong's Birthday",
    "2024-06-17": "Hari Raya Haji",
    "2024-07-07": "Awal Muharram",
    "2024-08-31": "National Day",
    "2024-09-16": "Malaysia Day",
    "2024-09-26": "Maulidur Rasul",
    "2024-11-01": "Deepavali",
    "2024-12-25": "Christmas Day",
    "2025-01-01": "New Year's Day",
    "2025-01-29": "Chinese New Year",
    "2025-01-30": "Chinese New Year Holiday",
    "2025-02-01": "Federal Territory Day",
    "2025-03-31": "Hari Raya Aidilfitri",
    "2025-04-01": "Hari Raya Aidilfitri Holiday",
    "2025-05-01": "Labour Day",
    "2025-05-12": "Wesak Day",
    "2025-06-02": "Agong's Birthday",
    "2025-06-06": "Hari Raya Haji",
    "2025-06-27": "Awal Muharram",
    "2025-08-31": "National Day",
    "2025-09-05": "Maulidur Rasul",
    "2025-09-16": "Malaysia Day",
    "2025-10-20": "Deepavali",
    "2025-12-25": "Christmas Day",
    "2026-01-01": "New Year's Day",
    "2026-02-01": "Federal Territory Day",
    "2026-02-17": "Chinese New Year",
    "2026-02-18": "Chinese New Year Holiday",
    "2026-03-20": "Hari Raya Aidilfitri",
    "2026-03-21": "Hari Raya Aidilfitri Holiday",
    "2026-05-01": "Labour Day",
    "2026-05-27": "Hari Raya Haji",
    "2026-05-31": "Wesak Day",
    "2026-06-01": "Agong's Birthday",
    "2026-08-31": "National Day",
    "2026-09-16": "Malaysia Day",
    "2026-11-08": "Deepavali",
    "2026-12-25": "Christmas Day",
}

HOLIDAY_IMPACT: dict[str, int] = {
    "Hari Raya Aidilfitri": -45,
    "Hari Raya Aidilfitri Holiday": -45,
    "Chinese New Year": -35,
    "Chinese New Year Holiday": -40,
    "Hari Raya Haji": -30,
    "Wesak Day": -25,
    "Deepavali": -25,
    "Labour Day": -20,
    "Agong's Birthday": -20,
    "Malaysia Day": -20,
    "National Day": -15,
    "Awal Muharram": -15,
    "Maulidur Rasul": -15,
    "Federal Territory Day": -15,
    "Thaipusam": -15,
    "Christmas Day": -20,
    "New Year's Day": -10,
}

FEATURE_COLS = [
    "station", "year", "month", "day", "day_of_week", "is_weekend",
    "is_holiday", "holiday_name",
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
]
CAT_FEATURES = ["station", "holiday_name"]


def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    date_str = df["date"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = date_str.isin(MY_HOLIDAYS).astype(int)
    df["holiday_name"] = date_str.map(MY_HOLIDAYS).fillna("None")
    return df


def engineer_features(
    station_daily: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add lag/rolling features and split train/test."""
    lag_days: list[int] = parameters["feature_engineering"]["lag_days"]
    rolling_windows: list[int] = parameters["feature_engineering"]["rolling_windows"]
    train_cutoff = pd.to_datetime(parameters["feature_engineering"]["train_cutoff"])

    df = station_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = _add_holiday_features(df)

    grp = df.groupby("station")["ridership"]
    for lag in lag_days:
        df[f"lag_{lag}"] = grp.shift(lag)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = grp.shift(1).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )

    df = df.dropna(subset=[f"lag_{lag_days[-1]}"])

    train = df[df["date"] <= train_cutoff].copy()
    test = df[df["date"] > train_cutoff].copy()

    return train, test


def train_model(
    features_train: pd.DataFrame,
    parameters: dict,
) -> CatBoostRegressor:
    """Train CatBoost on station_daily departure forecasting."""
    p = parameters["training"]
    target = p["target_col"]

    X = features_train[FEATURE_COLS]
    y = features_train[target]

    cat_idx = [FEATURE_COLS.index(c) for c in CAT_FEATURES]

    model = CatBoostRegressor(
        iterations=p["iterations"],
        learning_rate=p["learning_rate"],
        depth=p["depth"],
        random_seed=p["random_seed"],
        cat_features=cat_idx,
        verbose=100,
        loss_function="RMSE",
    )
    model.fit(X, y)
    return model


def evaluate_and_save_actuals(
    features_test: pd.DataFrame,
    model: CatBoostRegressor,
    parameters: dict,
) -> pd.DataFrame:
    """Predict on test set, log metrics, return actuals parquet (date col = 'date')."""
    target = parameters["training"]["target_col"]

    X_test = features_test[FEATURE_COLS]
    y_test = features_test[target]

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1))) * 100

    print(f"Test MAE:  {mae:.1f}")
    print(f"Test RMSE: {rmse:.1f}")
    print(f"Test MAPE: {mape:.2f}%")

    actuals = features_test[["station", "date", target]].copy()
    actuals = actuals.rename(columns={target: "ridership"})
    return actuals
