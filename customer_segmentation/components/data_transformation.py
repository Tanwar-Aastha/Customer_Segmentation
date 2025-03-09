import os
import sys
from customer_segmentation.logging.logger import logging
from customer_segmentation.exception.exception import CustomerSegmentationException
from customer_segmentation.utils import save_objects_file
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time
import nltk
import string
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import WordNetLemmatizer,PorterStemmer,wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler,MaxAbsScaler,FunctionTransformer


class CustomFunctionsClass:

    def cat_function(cat_data):
        cat_data['Product']=np.where(cat_data['Product'].isin(['Money transfers','Payday loan','Other financial service','Prepaid card','Virtual currency']),'Other currency',cat_data['Product'])
        return cat_data
    
    def nlp_function(nlp_data):
        nlp_data = nlp_data['Issue'] 
        tokenized_data = nlp_data.apply(lambda x: wordpunct_tokenize(x.lower()))# tokenizing the data

        def remove_punctuation(text):
            return [w for w in text if w not in string.punctuation]
        no_punctuation_data = tokenized_data.apply( lambda x: remove_punctuation(x))# removing punctuations from tokenized data

        Stop_words = stopwords.words('english')
        Removed_Stopwords = [w for w in no_punctuation_data if not w in Stop_words]
        Removed_Stopwords = pd.Series(Removed_Stopwords)# removing stopwords

        def lemmatize_text(text):
            lem_text = [WordNetLemmatizer().lemmatize(w,pos = 'v') for w in text]
            return lem_text
        lemmatized_data = Removed_Stopwords.apply(lambda x:lemmatize_text(x)) #lemmatizing the data

        def stem_text(text):
            stem_text = [PorterStemmer().stem(w) for w in text]
            return stem_text
        stemmed_data = lemmatized_data.apply(lambda x:stem_text(x))# stemming the lemmatized data

        cleaned_data=[" ".join(x) for x in stemmed_data]
        logging.info("Text preprocessing succesfully done")
        logging.info("Returnnig preprocessed data")
        return cleaned_data
    
    def date_time_function(datetime_data):
        datetime_data['Date received']=pd.to_datetime(datetime_data['Date received'])
        datetime_data['Date sent to company']=pd.to_datetime(datetime_data['Date sent to company'])
        datetime_data['days_held']=(datetime_data['Date sent to company']-datetime_data['Date received']).dt.days
        return datetime_data['days_held'].values.reshape(-1,1)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")# path for storing pickle file

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
     
    def get_data_transformation_obj(self):
        """
        This function is responsible for data transformation
        """
        try:
            start=time.time()
            logging.info("Data transformation started")
            
            Categorical_Column=['Product','Timely response?','Company response to consumer','Submitted via']
            NLP_Column=['Issue']
            Date_Time_Column=['Date sent to company','Date received']

            logging.info("Categorical pipeline initiated")
            Cat_pipeline=Pipeline( steps=[ #categorical pipeline
                                         ("Categorical_function",FunctionTransformer(CustomFunctionsClass.cat_function,validate=False))
                                        ,("Categorical_Imputer",SimpleImputer(strategy='most_frequent'))
                                        ,('Categorical_Onehot',OneHotEncoder(sparse_output=False,drop='first'))
                                        ,("Categorical_Scaler",MinMaxScaler())
                                        ]) 
            logging.info("Nlp pipeline initiated")
            Nlp_pipline=Pipeline( steps=[#NLP pipeline
                                        ("Nlp_extration",FunctionTransformer(CustomFunctionsClass.nlp_function,validate=False))
                                        ,("CountVector",CountVectorizer())
                                        ,("NLP_Scaler",MaxAbsScaler())
                                        ])
            logging.info("Date_time pipeline initiated")
            Date_time_pipeline=Pipeline(steps=[#Datetime pipeline
                                        ("Date_time_transformer",FunctionTransformer(CustomFunctionsClass.date_time_function,validate=False))
                                        ,("Date_time_Scaler",MinMaxScaler())
                                        ])

            logging.info("ColumnTransformer started running pipelines")    
            # Column Transformer for pipeline
            column_preprocessor=ColumnTransformer( 
                                        [
                                        ("Categorical_Transformer",Cat_pipeline,Categorical_Column)
                                        ,('NLP_Transformer',Nlp_pipline,NLP_Column)
                                        ,("Date_time_Transfomer",Date_time_pipeline,Date_Time_Column)
                                        ]
                                        ,remainder='passthrough')# untoched column that are not transformed   
            logging.info("ColumnTransformer initiated pipelines succesfully") 
            end=time.time()
            logging.info("Data transformation initited succesfully in: {:.2f} seconds".format(end - start))
            return column_preprocessor

        except Exception as e:
            logging.error(str(e))
            raise CustomerSegmentationException(e,sys) 

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformation_obj()
            target_column_name = "Consumer disputed?"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on the training and testing dataframe.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Dataset transformation succesfully done transforming our dataset")

            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()

            Label_encoder_object = LabelEncoder()
            logging.info("Applying label encoder on the trainin and testing target feature")

            target_feature_train_arr = Label_encoder_object.fit_transform(target_feature_train_df)
            target_feature_test_arr = Label_encoder_object.transform(target_feature_test_df)
            logging.info("Target column transforming successfully done.")

            print("Shape of input_feature_train_arr:", input_feature_train_arr.shape)
            print("Shape of target_feature_train_arr:", target_feature_train_arr.shape)


            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr.reshape(-1, 1)]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr.reshape(-1, 1)]

            # Saving the preprocessor pickle file
            save_objects_file(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessor_obj
            )

            return (train_arr, 
                    test_arr, 
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.error(str(e))
            raise CustomerSegmentationException(e,sys) 

    