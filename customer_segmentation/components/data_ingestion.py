import os
import sys
from customer_segmentation.logger import logger
from customer_segmentation.exception import exception
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import train_test_split


@dataclass
class DataIngestionConfig:
    """
    Defining the path for train, test and raw data
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        """Creating an object for the DataIngestionConfig class"""
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logger.logger.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("D:\Aastha\Projects\Customer_dispute_Segmentation\Customer_data\data.csv")
            logger.logger.info("Read the Data as a DataFrame")

            # creating the artifact directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # loading the data into raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.logger.info("Data uploaded in raw data path")


        except:
            pass
