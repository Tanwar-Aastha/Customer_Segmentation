import os
import sys
import pickle
from customer_segmentation.exception.exception import CustomerSegmentationException
from customer_segmentation.logging.logger import logging
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import accuracy_score


def save_objects_file(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info("Object directory initiated")
        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)
        logging.info("File successfuly saved at {}".format(file_path))  

    except Exception as e:
        logging.error(str(e))
        raise CustomerSegmentationException(e, sys)

def load_obj_file(file_path):
    try:
        logging.info("Loading pickel file of {}".format(file_path))
        with open(file_path,"rb") as file_object:
            return pickle.load(file_object)
        
    except Exception as e:
        logging.error(str(e))
        raise CustomerSegmentationException(e, sys)   
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    """
    This function will return a dictionary of models with their accuracy score as values.
    """
    try:
        report = {}
        logging.info("Starting the model fitting and accuracy score calculation")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info("Predicted the y test")

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            logging.info(f"Calculated the accuracy score for {list(models.keys())[i]} model")

            report[list(models.keys())[i]] = test_model_score

        return report



    except Exception as e:
        logging.error(str(e))
        raise CustomerSegmentationException(e, sys)
   
