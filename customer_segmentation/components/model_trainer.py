import os
import sys
from dataclasses import dataclass
from customer_segmentation.logging.logger import logging
from customer_segmentation.exception.exception import CustomerSegmentationException
from customer_segmentation.utils import save_objects_file, evaluate_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

# Class responsible for model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Entered the model trainer component
        """
        try:
            logging.info("Splitting the training and testing input")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            models = {
                "Random Forest":RandomForestClassifier()
            }

            params = {
                    "Random Forest":{
                        "n_estimators": [8,16,32,64,128,256]
                    }
                }


            logging.info("Starting model evaluations")
            model_report:dict= evaluate_model(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, models=models, params=params)

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

            save_objects_file(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            
            # checking the accuracy of the best model
            prediction  = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, prediction)
            return accuracy

        except Exception as e:
            raise CustomerSegmentationException(e, sys)
