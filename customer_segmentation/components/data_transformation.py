import os
import sys
from customer_segmentation.logging.logger import logging
from customer_segmentation.exception.exception import CustomerSegmentationException
from customer_segmentation.utils import save_object_file
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time
import nltk
import string
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import WordNetLemmatizer,wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler,MaxAbsScaler,FunctionTransformer


# Creating a Custom Function class
class CustomFunction:
    def cat_function(cat_data):
        cat_data['Product']=np.where(cat_data['Product'].isin([
            'Money transfers',
            'Payday loan','Other financial service',
            'Prepaid card','Virtual currency']),
                                     'Other currency',
                                     cat_data['Product'])
        return cat_data
    
    def nlp_function(nlp_data):
        nlp_data = nlp_data['Issue']
        # tokenizing the data
        tokenize_data = nlp_data.apply(lambda x: wordpunct_tokenize(x.lower()))

        def remove_punctuation(text):
            """This function will remove punctuations from the words present in the text."""
            return [word for word in text if word not in string.punctuation]
        
        # removing punctuations from the tokenize text
        no_punct_data = tokenize_data.apply(lambda x: remove_punctuation(x))

        stop_words = stopwords.words('english')
        removed_stopwords = [word for word in no_punct_data if word not in stop_words]
        removed_stopwords = pd.Series(removed_stopwords)

        def lemmatize_text(text):
            """This function will lemmatize the text"""
            lem_text = [WordNetLemmatizer().lemmatize(word, pos='v') for word in text]
            return lem_text
        
        lemmatize_data = removed_stopwords.apply(lambda x: lemmatize_text(x))

        cleaned_data = [" ".join(x) for x in lemmatize_data]
        logging.info("Text Preprocessing successfully done")
        logging.info("returning the Preprocessed data")
        return cleaned_data
    
    def date_time_function(datetime_data):
        datetime_data['Date received'] = pd.to_datetime(datetime_data['Date received'])
        datetime_data['Date sent to company']=pd.to_datetime(datetime_data['Date sent to company'])
        datetime_data['days_held']=(datetime_data['Date sent to company']-datetime_data['Date received']).dt.days
        return datetime_data['days_held'].values.reshape(-1,1)


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        """This function is responsible for data transformation"""
        try:
            Categorical_col = ['Product','Timely response?','Company response to consumer','Submitted via']
            NLP_col = ['Issue']
            Date_Time_col = ['Date sent to company','Date received']

            logging.info("Starting Categorical pipeline")
            cat_pipeline = Pipeline(steps=[
                ("Categorical_function", FunctionTransformer(CustomFunction.cat_function, validate=False)),
                ("Categorical Imputer", SimpleImputer(strategy='most_frequent')),
                ("Categorical_Onehot", OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')),
                ("Categorical_Scaler", MinMaxScaler())
            ])

            logging.info("Starting NLP Pipeline")
            nlp_pipeline = Pipeline(steps=[
                ("nlp_extraction", FunctionTransformer(CustomFunction.nlp_function, validate=False)),
                ("CountVector", CountVectorizer()),
                ("nlp_Scaler", MaxAbsScaler())
            ])

            logging.info("Starting Date Time Pipeline")
            date_time_pipeline = Pipeline(steps=[
                ("Date_time_transformer", FunctionTransformer(CustomFunction.date_time_function, validate=False)),
                ("Date_time_Scaler", MinMaxScaler())
            ])


            # Column Transformer
            logging.info("Starting Column transformation")
            preprocessor = ColumnTransformer(transformers=[
                ("Cat_Transformer", cat_pipeline, Categorical_col),
                ("NLP_transformer", nlp_pipeline, NLP_col),
                ("Date_time_Transformer", date_time_pipeline, Date_Time_col)
            ], remainder='passthrough')

            return preprocessor
        
        except Exception as e:
            raise CustomerSegmentationException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating Data Transformation")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformation_obj()
            target_col = 'Consumer disputed?'

            logging.info("Setting the dependent and independent features")
            train_features = train_data.drop(columns=[target_col], axis=1)
            train_target_feature = train_data[target_col]

            test_features = test_data.drop(columns=[target_col], axis=1)
            test_target_feature = test_data[target_col]

            Label_encoder_obj = LabelEncoder()

            logging.info("Transforming the independent train and test features")
            train_features_arr = preprocessor_obj.fit_transform(train_features)
            test_features_arr = preprocessor_obj.transform(test_features)

            train_features_arr = train_features_arr.toarray()
            test_features_arr = test_features_arr.toarray()

            logging.info("Transforming the target train and test feature")
            train_target_feature_arr = Label_encoder_obj.fit_transform(train_target_feature)
            test_target_feature_arr = Label_encoder_obj.transform(test_target_feature)
            logging.info("Data Transformation successfully Done")

            train_arr = np.c_[train_features_arr, train_target_feature_arr]
            test_arr = np.c_[test_features_arr, test_target_feature_arr]

            logging.info("Saving the Preprcessor file")
            save_object_file(file_path=self.data_transformation_config.preprocessor_path,
                             object=preprocessor_obj)
            
            return (train_arr, test_arr)

        except Exception as e:
            raise CustomerSegmentationException(e, sys)