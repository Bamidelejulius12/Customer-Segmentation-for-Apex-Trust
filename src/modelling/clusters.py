# cluster.py
import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
from src.data.data_processing import data_processing
from utils.mlflow_config import setup_mlflow
from utils.model_loader import load_model_and_params

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class clustering_engine:

    def __init__(self):
        # training data pipeline - returns scaled data, original data, and scaler
        rfm_data_scaled, rfm_data, scaler = data_processing()

        self.rfm_scaled_df = rfm_data_scaled  
        self.rfm_data = rfm_data  
        self.scaler = scaler  

        self.model = None
        self.mlflow_setup = setup_mlflow()

    def find_optimal_clusters(self):
        try:
            silhouette_scores = []
            cluster_range = list(range(2, 11))
            
            best_score = -1
            best_model = None

            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.rfm_scaled_df)

                score = silhouette_score(self.rfm_scaled_df, labels)
                silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_model = kmeans

            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
            
            self.model = best_model  # Store the best model
            self.optimal_k = optimal_k  # Store optimal K for later use

            logging.info(f"Optimal K: {optimal_k}")
            logging.info(f"Best silhouette: {best_silhouette}")

            return optimal_k, best_silhouette  # Only return what's needed

        except Exception as e:
            logging.error(f"Error finding optimal clusters: {e}")
            raise

    # TRAIN KMeans model and LOG TO MLFLOW
    def train_and_log_model(self):
        try:
            self.mlflow_setup

            with mlflow.start_run():

                optimal_k, best_silhouette = self.find_optimal_clusters()

                # Log parameters and metrics to MLflow
                mlflow.log_param("optimal_k", optimal_k)
                mlflow.log_metric("silhouette_score", best_silhouette)

                # Log the KMeans model (already stored in self.model)
                mlflow.sklearn.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="Apex_Trust_model"
                )

                logging.info("Model logged and registered successfully to MLflow")

        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

    # INFERENCE
    def apply_clustering(self):
        """
        Load pre-trained model and optimal K from MLflow and apply clustering
        """
        try:
            self.mlflow_setup

            # Load the pre-trained model and optimal K
            model, optimal_k = load_model_and_params()

            print(f"\n=== APPLYING K-MEANS WITH {optimal_k} CLUSTERS ===")
            
            # Apply clustering using the loaded model and optimal K
            self.rfm_data['Cluster'] = model.predict(self.rfm_scaled_df)
            
            print(self.rfm_data.head())
            
            logging.info("Clustering applied successfully...")

            return self.rfm_data

        except Exception as e:
            logging.error(f"Error in apply_clustering: {e}")
            raise




# if __name__ == "__main__":
#     cluster_engine = clustering_engine()
#     #cluster_engine.train_and_log_model()
#     clustered_rfm_df = cluster_engine.apply_clustering()