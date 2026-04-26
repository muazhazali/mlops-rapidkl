from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_od_daily, create_station_daily


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_station_daily,
            inputs=["raw_2023", "raw_2024", "raw_2025", "raw_2026"],
            outputs="station_daily",
            name="create_station_daily_node",
        ),
        node(
            func=create_od_daily,
            inputs=["raw_2023", "raw_2024", "raw_2025", "raw_2026"],
            outputs="od_daily",
            name="create_od_daily_node",
        ),
    ])
