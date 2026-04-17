import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.data_ingestion import data_ingestion
from config.constant import Input_Data_file_path
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def data_validation(data = pd.DataFrame):
    try:
        transaction_data = data.copy()
        duplicates = transaction_data.duplicated().sum()
        logging.info(f"we have the total number of {duplicates} in the dataset")
        logging.info(f"checking for duplicated transaction id...")
        duplicated_transaction_id = transaction_data["TransactionID"].duplicated().sum()
        logging.info(f"we have the total number of {duplicated_transaction_id} duplicated transaction ID")
        transaction_data = transaction_data.drop_duplicates(subset=["TransactionID"])
        logging.info(f"successfully dropped duplicated transaction ID....")
        logging.info("checking for the number of unique customer that we have..")
        unique_customer = transaction_data["CustomerID"].nunique()
        logging.info(f"we have the total number of {unique_customer} customer in the dataset")
        logging.info(f"checking for the dataset information...")
        logging.info(f"these are the data and their data types {transaction_data.info()}")
        transaction_data["TransactionDate"] = pd.to_datetime(transaction_data["TransactionDate"], errors = "coerce")
        logging.info(f"date time successfully transformed...")
        logging.info(f"we have the total number of missing values {transaction_data.isna().sum()}")
        transaction_data = transaction_data.dropna()
        logging.info(f"missing values sucessfully dropped")
        logging.info(f"this is total number of dataset {transaction_data.shape}")
        logging.info(f"we have a total number of {transaction_data['CustomerID'].nunique()} customer")
        return transaction_data
    except Exception as e:
        logging.error(f"error occurred while validating data {e}")

