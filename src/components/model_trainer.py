import os 
import sys
import pandas as pd
from dataclasses import dataclass
from src.logging import logging
from src.exception import CustomerSegmentationException
from src.utils import evaluate_model, save_object_file

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_model_path= os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the data into training and test input")
            X_train, y_train, X_test, y_test = (train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1])
            
            models = {
                "Decision Tree":DecisionTreeClassifier()
            }
            params = {
                "Decision Tree":{
                    "criterion":['gini','entropy'],
                    "max_depth": [5, 10, 20, 30, 40],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [5, 10, 20]
                }}

            logging.info("Starting model evaluation")
            model_report:dict = evaluate_model(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, 
                                               models=models, params=params)

            logging.info("Sorting model based on their accuracy score")
            # Getting the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Getting the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Setting up a threshold
            if best_model_score < 0.6:
                raise CustomerSegmentationException("No best Model found")
            logging.info("Best model found in both training and test set")


            logging.info("Saving the best model pickle file")
            save_object_file(
                file_path=self.model_trainer_config.trained_model_path,
                object=best_model
            )
            
            # checking the accuracy of the best model
            prediction  = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, prediction)
            return accuracy


        except Exception as e:
            raise CustomerSegmentationException(e, sys)