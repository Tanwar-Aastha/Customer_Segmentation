import os
import sys
import pickle
from customer_segmentation.exception.exception import CustomerSegmentationException
from customer_segmentation.logging.logger import logging


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
    
   
