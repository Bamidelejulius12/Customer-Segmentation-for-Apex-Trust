import mlflow
from mlflow.tracking import MlflowClient
from utils.mlflow_config import setup_mlflow


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


if __name__ == "__main__":
    load_model_and_params()