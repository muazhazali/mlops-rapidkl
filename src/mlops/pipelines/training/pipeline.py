from kedro.pipeline import Pipeline, node, pipeline

from .nodes import engineer_features, evaluate_and_save_actuals, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=engineer_features,
            inputs=["station_daily", "parameters"],
            outputs=["features_train", "features_test"],
            name="engineer_features_node",
        ),
        node(
            func=train_model,
            inputs=["features_train", "parameters"],
            outputs="catboost_model",
            name="train_model_node",
        ),
        node(
            func=evaluate_and_save_actuals,
            inputs=["features_test", "catboost_model", "parameters"],
            outputs="actuals",
            name="evaluate_node",
        ),
    ])
