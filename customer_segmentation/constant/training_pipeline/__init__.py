import os
import sys
import pandas as pd
import numpy as np

"""
Defining common constant variables for training pipeline
"""
PIPELINE_NAME: str = "customer_segmentation"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


"""
Data ingestion related constants start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2