from src.modelling.clustering import clustering_engine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from utils.cluster_utils import cluster_analyzer, assign_cluster_names, cluster_grouping
from src.modelling.clustering import clustering_engine

import logging

logging.basicConfig(
    level= logging.DEBUG,
    format= "%(asctime)s - %(levelname)s - %(message)s"
)

class segment_engine:

    def cluster_grouper():
        try:
            cluster_engine = clustering_engine()
            clustered_rfm_df = cluster_engine.apply_clustering()
            cluster_analysis, rfm_data = cluster_analyzer(clustered_rfm_df)
            segmented_rfm_data = assign_cluster_names(rfm_data, cluster_analysis)
            customer_segment_data = cluster_grouping(segmented_rfm_data)
            print(segmented_rfm_data.head().T)
            print(customer_segment_data.head().T)
            logging.info(f"Data successfully clustered & segmented")
            return segmented_rfm_data, customer_segment_data
        except Exception as e:
            logging.error(f"error occurred while clustering and segmenting the data {e}")

def main():
    engine = segment_engine
    engine.cluster_grouper()

main()