import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from src.data.data_processing import data_processing

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class clustering_engine:
    def __init__(self):
        rfm_data_scaled, rfm_data = data_processing()
        self.rfm_scaled_df = rfm_data_scaled
        self.rfm_data = rfm_data
        self.optimal_k = 4
    
    def find_optimal_clusters(self):
        try:
            """Determine optimal number of clusters"""
            
            wcss = []
            silhouette_scores = []
            cluster_range = list(range(2, 11))

            for k in cluster_range:
                kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_model.fit(self.rfm_scaled_df)
                wcss.append(kmeans_model.inertia_)
                if k > 1:
                    score = silhouette_score(self.rfm_scaled_df, kmeans_model.labels_)
                    silhouette_scores.append(score) 

            self.optimal_k = cluster_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
            logging.info(f"the optimal k  is {self.optimal_k}")
            logging.info(f"the best silhoutte_score is {best_silhouette}")

            return silhouette_scores, self.optimal_k, best_silhouette

        except Exception as e:
            logging.error(f"error occured while finding optimal clusters...{e}")

    def apply_clustering(self):
        """Apply K-means clustering cleanly without global"""
        try:
            if self.optimal_k is None:
                _, self.optimal_k, _ = self.find_optimal_clusters()  

            kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            self.rfm_data['Cluster'] = kmeans.fit_predict(self.rfm_scaled_df)
            self.rfm_data.head(5)
            logging.info(f"clusters suceessfully applied")
            return self.rfm_data
        except Exception as e:
            logging.error(f"error occured while applying clusters.. {e}")

