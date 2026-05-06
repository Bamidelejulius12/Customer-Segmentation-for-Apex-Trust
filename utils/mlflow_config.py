import dagshub
import mlflow
from dotenv import load_dotenv
import os
import mlflow

load_dotenv(override=True)

# def setup_mlflow():
#     dagshub.init(
#         repo_owner="babatundejulius911",
#         repo_name="Customer-Segmentation-for-Apex-Trust",
#         mlflow=True
#     )

#     mlflow.set_experiment("customer_segmentation")

def setup_mlflow():

    dagshub_token = os.getenv("MFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Shop_env_DAGSHUB_TOKEN is not set")

    # Set MLflow auth (DagsHub uses token as both username & password)
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    repo_owner = "babatundejulius911"
    repo_name = "Customer-Segmentation-for-Apex-Trust"

    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("customer_segmentation")
    