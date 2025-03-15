import os
import sys
import pandas as pd
from src.logging import logging
from src.exception import CustomerSegmentationException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_path = os.path.join('artifacts', "train.csv")
    test_path = os.path.join('artifacts',"test.csv")
    raw_data_path = os.path.join('artifacts', "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Get the base directory (project root, not src)
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gives 'src/components'
            PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Moves up two levels to project root

            csv_path = os.path.join(PROJECT_ROOT, "dataset", "modified_data.csv")
            df = pd.read_csv(csv_path)

            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            # loading the data into raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Data uploaded in raw data path")

            # Splitting the data into train test set
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data has been split into train test set")

            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info("Ingestion is completed")

            return(
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )

        except Exception as e:
            raise CustomerSegmentationException(e, sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    # data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))