import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format= "%(asctime)s - %(levelname)s - %(message)s"
)

def cluster_analyzer(rfm_data: pd.DataFrame):
    try:
        cluster_analysis = rfm_data.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Recency_Score': 'mean',
            'Frequency_Score': 'mean',
            'Monetary_Score': 'mean',
            'CustomerID': 'count',
            'CustAccountBalance': 'mean',
            'CustGender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)

        cluster_analysis.columns = [
            'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
            'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score',
            'Count', 'Avg_Account_Balance', 'Most_Common_Gender'
        ]
        logging.info("clusters has been analyzed successfully.")
        return cluster_analysis, rfm_data
    except Exception as e:
        logging.error(f"error occurred while analyzing clusters {e}")




def assign_cluster_name(stats):
    recency = stats['Avg_Recency']
    frequency = stats['Avg_Frequency']
    monetary = stats['Avg_Monetary']

    if recency > 365:
        if monetary > 20000:
            return "High-Value Dormant Customers"
        else:
            return "Dormant Low-Value Customers"

    elif recency > 180:
        if monetary > 20000:
            return "High-Value Inactive Customers"
        else:
            return "Inactive Low-Value Customers"

    if frequency > 10 and monetary > 30000:
        return "Active VIP Customers"
    elif frequency > 8 and monetary < 10000:
        return "Active Low-Value Customers"
    else:
        return "Regular Customers"


def assign_cluster_names(rfm_data, cluster_analysis):
    try:
        rfm_data['Cluster_Name'] = rfm_data['Cluster'].map(
            lambda x: assign_cluster_name(cluster_analysis.loc[x])
        )
        logging.info(f"cluster names assigned successfully...")
        return rfm_data
    except Exception as e:
        logging.error(f"error occurred while assigning names to clusters {e}")

def cluster_grouping(rfm_data: pd.DataFrame):
    try:
        # --- Generate cluster profiles from rfm_df ---
        cluster_profiles = rfm_data.groupby('Cluster_Name').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count',
            'CustAccountBalance': 'mean'
        }).round(2)

        cluster_profiles = cluster_profiles.rename(columns={
            'Recency': 'Avg_Recency',
            'Frequency': 'Avg_Frequency',
            'Monetary': 'Avg_Monetary',
            'CustomerID': 'Customer_Count',
            'CustAccountBalance': 'Avg_Account_Balance'
        })
        cluster_profiles['Percentage'] = (
            cluster_profiles['Customer_Count'] / len(rfm_data) * 100
        ).round(2)
        logging.info(f"customer have grouped successfully... ")

        return cluster_profiles
    except Exception as e:
        logging.error(f"error occurred while grouping the customer data {e}")