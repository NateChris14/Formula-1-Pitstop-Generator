import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import RareLabelEncoder
from feature_engine.selection import DropDuplicateFeatures

import os
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            #Defining numerical and categorical columns
            numeric_columns = ["Grid Position", "Final Position", "Fastest Lap Time", "Points", "Number of Laps", 
                   "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Rain (mm)"]
            low_cardinality_cat = ["Status", "Weather"]
            high_cardinality_cat = ["Race Name","Driver ID", "Constructor Name"]

            #Numerical Pipeline
            num_transformer = Pipeline(steps=[
            ("drop_duplicates",DropDuplicateFeatures()),
            ("Imputer",SimpleImputer(strategy='median')),
            ("outlier_removal",Winsorizer(capping_method='gaussian',tail='both',fold=2.0)),
            ("scaler",RobustScaler())
            ])

            # Low-Cardinality Categorical Pipeline (One-Hot Encoding)
            low_cardinality_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("rare_label", RareLabelEncoder(tol=0.05, replace_with="Other")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))  # Ensures all categories are numeric
            ])

            # High-Cardinality Categorical Pipeline (Ordinal Encoding)
            high_cardinality_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("rare_label", RareLabelEncoder(tol=0.05, replace_with="Other")),
            ("ordinal", OrdinalEncoder())  # Converts categories to numeric values
            ])

            logging.info(f"Numerical Columns:{numeric_columns}")
            logging.info(f"Low cardinality categorical features: {low_cardinality_cat}")
            logging.info(f"High cardinality categorical features: {high_cardinality_cat}")

            preprocessor = ColumnTransformer(transformers=[
            ("num",num_transformer,numeric_columns),
            ("low_card_cat",low_cardinality_transformer,low_cardinality_cat),
            ("high_card_cat",high_cardinality_transformer,high_cardinality_cat)
            ],remainder='drop')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed!")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name = "Pit Stops"
            
            input_feature_train_df=train_df.drop([target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test df")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
