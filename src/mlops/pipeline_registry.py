"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from mlops.pipelines.data_processing.pipeline import create_pipeline as dp_pipeline
from mlops.pipelines.training.pipeline import create_pipeline as training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_proc = dp_pipeline()
    training = training_pipeline()

    return {
        "data_processing": data_proc,
        "training": training,
        "__default__": data_proc + training,
    }
