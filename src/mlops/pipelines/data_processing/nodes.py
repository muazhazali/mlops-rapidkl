from __future__ import annotations

import pandas as pd


def merge_raw(
    raw_2023: pd.DataFrame,
    raw_2024: pd.DataFrame,
    raw_2025: pd.DataFrame,
    raw_2026: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge raw OD CSVs, produce station_daily and od_daily."""
    df = pd.concat([raw_2023, raw_2024, raw_2025, raw_2026], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # OD daily: station-to-station only (exclude A0 aggregates)
    od = df[
        (df["origin"] != "A0: All Stations")
        & (df["destination"] != "A0: All Stations")
    ].copy()

    # Station daily: total departures per origin per day
    station_daily = (
        od.groupby(["origin", "date"], as_index=False)["ridership"]
        .sum()
        .rename(columns={"origin": "station"})
    )

    return station_daily, od


def create_station_daily(
    raw_2023: pd.DataFrame,
    raw_2024: pd.DataFrame,
    raw_2025: pd.DataFrame,
    raw_2026: pd.DataFrame,
) -> pd.DataFrame:
    station_daily, _ = merge_raw(raw_2023, raw_2024, raw_2025, raw_2026)
    return station_daily


def create_od_daily(
    raw_2023: pd.DataFrame,
    raw_2024: pd.DataFrame,
    raw_2025: pd.DataFrame,
    raw_2026: pd.DataFrame,
) -> pd.DataFrame:
    _, od = merge_raw(raw_2023, raw_2024, raw_2025, raw_2026)
    return od
