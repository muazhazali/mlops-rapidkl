# mlops-rapidkl

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

End-to-end MLOps pipeline for daily ridership forecasting across the KL Rapid Rail network (KJ LRT, AG LRT, Kajang MRT, Putrajaya MRT, Monorail). Predicts next-day station departures using CatBoost trained on 3 years of origin-destination data from [data.gov.my](https://data.gov.my/data-catalogue/ridership_od_rapidrail_daily).

## Architecture

```
data/01_raw/          ← OD ridership CSVs (2023–2026)
    ↓  [Kedro: data_processing]
data/02_intermediate/ ← station_daily.parquet + od_daily.parquet
    ↓  [Kedro: training]
data/05_model_input/  ← features_train.parquet + features_test.parquet
data/06_models/       ← catboost_model.pkl
data/07_model_output/ ← actuals.parquet + predictions.parquet
    ↓
entrypoints/inference.py  ← rolling replay (1 day/second)
entrypoints/app_ui.py     ← Dash dashboard on :8050
```

**Tech stack:** Kedro · CatBoost · pandas · Dash + Plotly + Bootstrap · Docker · uv

## Dashboard

- **Line → Station** drill-down in sidebar
- **4 metric cards:** today's actual, predicted value, 30-day MAPE, Malaysian holiday flag
- **Main chart:** predicted vs actual daily departures with forecast zone
- **Top destinations:** OD-derived bar chart for selected origin station
- **Day-of-week pattern:** 30-day average, weekday/weekend split

## Quick start

### 1. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Download raw data

Place the CSVs in `data/01_raw/`:
```
rapidrail_2023_daily.csv
rapidrail_2024_daily.csv
rapidrail_2025_daily.csv
rapidrail_2026_daily.csv
```

Source: https://data.gov.my/data-catalogue/ridership_od_rapidrail_daily

### 3. Train

```bash
python entrypoints/training.py
```

Runs the full Kedro pipeline: data processing → feature engineering → CatBoost training → test evaluation. Artifacts saved to `data/`.

### 4. Run inference + UI

In two separate terminals:

```bash
# Terminal 1 — rolling inference replay
python entrypoints/inference.py

# Terminal 2 — dashboard
python entrypoints/app_ui.py
```

Open http://localhost:8050

### 5. Docker (all-in-one)

```bash
# Step 1: train (run once)
docker compose --profile train up ml-train --build

# Step 2: inference + UI
docker compose up ml-inference app-ui --build
```

## Configuration

All parameters in `conf/base/parameters.yml`:

| Key | Description |
|-----|-------------|
| `feature_engineering.lag_days` | Lag feature windows (default: 7, 14, 30 days) |
| `feature_engineering.train_cutoff` | Train/test split date |
| `training.iterations` | CatBoost iterations |
| `pipeline_runner.inference_interval_seconds` | Replay speed (1 = 1 day/second) |
| `ui.default_lookback_days` | Chart default window |

## Project structure

```
conf/base/
    catalog.yml       ← Kedro data catalog
    parameters.yml    ← all tunable parameters
src/mlops/
    pipelines/
        data_processing/  ← merge CSVs → station_daily, od_daily
        training/         ← features, CatBoost, evaluation
src/app_ui/
    app.py            ← Dash layout + callbacks
    utils.py          ← chart builders, data loaders
entrypoints/
    training.py       ← run Kedro default pipeline
    inference.py      ← rolling replay loop
    app_ui.py         ← start Dash server
```
