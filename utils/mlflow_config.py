import dagshub
import mlflow

def setup_mlflow():
    dagshub.init(
        repo_owner="babatundejulius911",
        repo_name="Customer-Segmentation-for-Apex-Trust",
        mlflow=True
    )

    mlflow.set_experiment("customer_segmentation")

