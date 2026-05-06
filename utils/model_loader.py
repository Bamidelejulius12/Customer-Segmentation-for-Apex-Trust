import mlflow
from mlflow.tracking import MlflowClient
from utils.mlflow_config import setup_mlflow
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level= logging.DEBUG
)

def load_model_and_params():
    setup_mlflow()

    client = MlflowClient()

    model_name = "Apex_Trust_model"

    # Get latest model version
    latest_versions = client.get_latest_versions(
        name=model_name,
        stages=["None", "Staging", "Production"]
    )

    if not latest_versions:
        raise ValueError(f"No registered model found for {model_name}")

    # Pick the most recent version
    latest_version = max(latest_versions, key=lambda x: int(x.version))

    model_uri = f"models:/{model_name}/{latest_version.version}"

    # Load model from registry
    model = mlflow.sklearn.load_model(model_uri)

    # Get run linked to this model version
    run_id = latest_version.run_id
    run = client.get_run(run_id)

    optimal_k = int(run.data.params.get("optimal_k"))

    print(f"Loaded model: {model_name} v{latest_version.version}")
    print(f"Run ID: {run_id}")
    print(f"Optimal K: {optimal_k}")
    print(type(model))

    return model, optimal_k


def is_new_model_better(new_score, experiment_name="Default"):
    try:
       
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            # No experiment yet → treat as first run
            return True, None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.silhouette_score DESC"],
            max_results=1
        )

        if not runs:
            return True, None

        previous_best = runs[0].data.metrics.get("silhouette_score")

        if previous_best is None:
            return True, None

        return new_score > previous_best, previous_best

    except Exception as e:
        # Fail safe: allow logging instead of breaking pipeline
        logging.error(f"error occurred while loading the previous {e} ")
        return True, None