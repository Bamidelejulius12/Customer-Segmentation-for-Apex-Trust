import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data.data_ingestion import data_ingestion
from config.constant import Input_Data_file_path
from src.data.data_validation import data_validation
import logging

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

class feature_eng:
    def calculate_rfm_metrics(data: pd.DataFrame):
        try:
            logging.info("calculating rfm metrics...")
            # Creating a reference date...
            reference_date = data["TransactionDate"].max() + pd.Timedelta(days = 1)
            # Creating the RFM columns...
            rfm_df = data.groupby('CustomerID').agg({
                'TransactionDate': lambda x: (reference_date - x.max()).days,
                'TransactionID': 'count',
                'TransactionAmount': 'sum'
            }).reset_index()

            rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            # adding the customer demographics
            customer_demographics = data.groupby('CustomerID').agg({
                'CustGender': 'first',
                'CustLocation': 'first',
                'CustomerDOB': 'first',
                'CustAccountBalance': 'last'
            }).reset_index()

            rfm_data = rfm_df.merge(customer_demographics, on="CustomerID", how='left')
            logging.info("RFM Metrics for each customer have been successfully calculated...")
            print(rfm_data.head(5))
            return rfm_data
        except Exception as e:
            logging.error(f"error occurred while calculating the RFM metrics: {e}")
    
    def calculate_rfm_scores(data: pd.DataFrame):
        try:
            logging.info("calculating rfm scores...")
            data['Recency_Score'] = pd.qcut(data['Recency'], q = 5, labels = [5, 4, 3, 2, 1])
            data['Frequency_Score'] = pd.qcut(data['Frequency'], q = 5, labels = [1, 2, 3, 4, 5])
            data['Monetary_Score'] = pd.qcut(data['Monetary'], q = 5, labels = [1, 2, 3, 4, 5])
            data[['Recency_Score', 'Frequency_Score', 'Monetary_Score']] = data[['Recency_Score', 'Frequency_Score', 'Monetary_Score']].astype(int)

            data['RFM_Score'] = data['Recency_Score'] + data['Frequency_Score'] + data['Monetary_Score']
            logging.info(f"RFM Scores for each customer have been successfully calculated...")
            print(data.head(5))
            return data
        except Exception as e:
            logging.error(f"error occurred while calculating the RFM metrics for each customer: {e}")


