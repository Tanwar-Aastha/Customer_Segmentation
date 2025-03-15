import os
import sys
import pandas as pd
import numpy as np
from src.logging import logging
from src.exception import CustomerSegmentationException
from src.utils import load_object


class PredictPipelineClass:
    def __init__(self):
        pass

    def predict(self, features):
        logging.info("Inside the PredictPipeline.predict()")

        logging.info("Loading the preprocessor pickle file")
        preprocessor_path = os.path.join('artifacts',"preprocessor.pkl")
        logging.info("Loading the model pickle file")
        model_path = os.path.join('artifacts',"model.pkl")

        preprocessor = load_object(file_path=preprocessor_path)
        model = load_object(file_path=model_path)

        try:
            scaled_data = preprocessor.transform(features)
            result = model.predict(scaled_data)

        except Exception as e:
            raise CustomerSegmentationException(e, sys)
        
        if result == 1:
            return "Customer will Dispute"
        else:
            return "Customer will not Dispute"
        

class CustomDataClass:
    def __init__(self,Date_Received,Product_Name,Issue_Detail,Submitted_Via,
                 Date_sent_to_company,Company_Response,Timely_Response):
        self.Date_Received = Date_Received
        self.Product_Name = Product_Name
        self.Issue_Detail = Issue_Detail
        self.Submitted_Via = Submitted_Via
        self.Date_sent_to_company = Date_sent_to_company
        self.Company_Response = Company_Response
        self.Timely_Response = Timely_Response

    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    "Date received":[self.Date_Received],
                    "Product":[self.Product_Name],
                    "Issue":[self.Issue_Detail],
                    "Submitted via":[self.Submitted_Via],
                    "Date sent to company":[self.Date_sent_to_company],
                    "Company response to consumer":[self.Company_Response],
                    "Timely response?":[self.Timely_Response]
                }

                dataframe_data = pd.DataFrame(custom_data_input_dict)
                directory_path = os.path.join(os.getcwd(), "user_data")
                os.makedirs(directory_path, exist_ok=True)

                file_path = os.path.join(directory_path, "user_data.csv")
                with open(file_path, 'a', newline='') as file:
                    dataframe_data.to_csv(file, header=False, index=False)

                return dataframe_data  
            except Exception as e:
                raise CustomerSegmentationException(e, sys) 