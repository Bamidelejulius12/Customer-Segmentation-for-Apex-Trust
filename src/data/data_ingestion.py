import numpy as np
import matplotlib.pyplot as plt
import seaborn
import logging
import pandas as pd
import logging
from src.connections.mongodb_connection import MongoDBConnection
from config.constant import Input_Data_file_path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def data_ingestion():
    try:
        df = pd.read_csv(Input_Data_file_path)
        # Get collection and load data
        # data_connector =  MongoDBConnection()
        # collection = data_connector.get_collection()
        # df = pd.DataFrame(list(collection.find({})))
        
        # # Remove MongoDB's _id column
        # if '_id' in df.columns:
        #     df = df.drop('_id', axis=1)
        
        logging.info(f"Data successfully loaded with total of {len(df):,} transactions...")
        print(df.head())
        return df
        
    except Exception as e:
        logging.error(f"Error while loading data: {e}")
        return None


