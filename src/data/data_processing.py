import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.features.feature_engineering import feature_eng
from src.data.data_ingestion import data_ingestion
from src.data.data_validation import data_validation
from config.constant import Input_Data_file_path
import logging

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

def data_processor(data: pd.DataFrame):
    try:
        logging.info("data preprocessor activated...")
        scaler = StandardScaler()
        rfm_processed_data = data[['Recency', 'Frequency', 'Monetary']].copy()

        rfm_processed_data = np.log1p(rfm_processed_data)
        rfm_processed_data = scaler.fit_transform(rfm_processed_data)

        rfm_processed_data = pd.DataFrame(
            rfm_processed_data, 
            columns = ['Recency', 'Frequency', 'Monetary'],
            index = data.index
        )
        print(rfm_processed_data.head(5))
        logging.info(f"Data successfully preprocessed and scaled...")
        return rfm_processed_data, scaler
    except Exception as e:
        logging.error(f"error occurred while scaling and processing the dataset {e}")

def data_processing():
    data = data_ingestion()
    data = data_validation(data)
    features_engineer = feature_eng
    rfm_data = features_engineer.calculate_rfm_metrics(data)
    rfm_data = features_engineer.calculate_rfm_scores(rfm_data)
    rfm_data_scaled, scaler = data_processor(rfm_data)
    print(rfm_data)
    print(rfm_data_scaled)
    return rfm_data_scaled, rfm_data, scaler
