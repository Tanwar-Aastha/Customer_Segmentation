import os
import sys
from customer_segmentation.logging.logger import logging
from customer_segmentation.exception.exception import CustomerSegmentationException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from customer_segmentation.components.data_transformation import DataTransformation, DataTransformationConfig
from customer_segmentation.components.model_trainer import ModelTrainer, ModelTrainerConfig

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
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("D:/Aastha/Projects/Customer_dispute_Segmentation/Customer_data/data.csv")
            logging.info("Read the Data as a DataFrame")

            # creating the artifact directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # loading the data into raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Data uploaded in raw data path")

            # Splitting the data into train test set
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data has been split into train test set")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomerSegmentationException(e, sys)

# checking the ingestion file
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))