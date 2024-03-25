import os
import sys
import pickle
from customer_segmentation.exception import CustomerSegmentationException
from customer_segmentation.logging import logger


def save_objects_file(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logger.info("Object directory initiated")
        with open(file_path, "wb") as file:
            pickle.dump(object, file)
        logger.info("File successfuly saved at {}".format(file_path))  

    except Exception as e:
        logger.error(str(e))
        raise CustomerSegmentationException(e, sys.exc_info())

def load_obj_file(file_path):
    try:
        logger.info("Loading pickel file of {}".format(file_path))
        with open(file_path,"rb") as file_object:
            return pickle.load(file_object)
        
    except Exception as e:
        logger.error(str(e))
        raise CustomerSegmentationException(e, sys.exc_info())   
    
   
