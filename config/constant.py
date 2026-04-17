import os
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Input_Data_file_path = os.path.join(BASE_DIR, "Dataset", "Bank_Trust_Data.csv")


# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = "Apex_trust_customer_transactions"
MONGO_COLLECTION = "Apex_Customer"
