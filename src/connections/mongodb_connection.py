from pymongo.mongo_client import MongoClient
import logging
from config.constant import MONGO_URI, MONGO_DATABASE, MONGO_COLLECTION

logging.basicConfig(
    level=logging.DEBUG,
    format= "%(asctime)s - %(levelname)s - %(message)s"
)

class MongoDBConnection:
    """MongoDB connection handler"""

    def __init__(self):
        self.uri =  MONGO_URI 
        self.database_name =  MONGO_DATABASE
        self.client = None
        self.db = None

        
    def get_mongo_connection(self):
        """Get MongoDB connection"""
        self.client = MongoClient(self.uri)
        self.db = self.client[self.database_name]
        return self.client, self.db

    def get_collection(self):
        """Get the collection directly"""
        self.client = MongoClient(self.uri)
        return self.client[self.database_name][MONGO_COLLECTION]
   