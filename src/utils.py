import os
import sys
import dill
from src.logging import logging
from src.exception import CustomerSegmentationException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object_file(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        logging.info("Object Directory Initiated")
        with open(file_path, 'wb') as file_object:
            dill.dump(object, file_object)
            logging.info("File successfuly saved at {}".format(file_path))
    except Exception as e:
        raise CustomerSegmentationException(e, sys)


def load_object(file_path):
    try:
        logging.info("Loading pickle file")
        with open(file_path, 'rb') as file_object:
            return dill.load(file_object)
    except Exception as e:
        raise CustomerSegmentationException(e, sys)
    


def evaluate_model(x_train, x_test, y_train, y_test, models, params):
    try:
        report = {}
        logging.info("Starting the model fitting and accuracy score calculation")
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3, verbose=1, n_jobs=-1)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)
            y_train_pred= model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info("Predicted the y test")

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            logging.info(f"Calculated the accuracy score for {list(models.keys())[i]} model")

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomerSegmentationException(e, sys)
