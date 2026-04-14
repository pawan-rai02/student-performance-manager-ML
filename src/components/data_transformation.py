import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


'''
this module is responsible for data transformation
- It defines a DataTransformationConfig class to store the file path for the preprocessor object
- It defines a DataTransformation class to perform data transformation
- The get_data_transformer_object function creates a preprocessing object that applies imputation and scaling to numerical columns and imputation, one-hot encoding, and scaling to categorical columns
- The initiate_data_transformation function reads the train and test data, applies the preprocessing object to
'''

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
         - Numerical columns: Imputation (median) and Scaling (StandardScaler)
         - Categorical columns: Imputation (most_frequent), OneHotEncoding and Scaling (StandardScaler with_mean=False)
         - ColumnTransformer is used to apply the transformations to the respective columns
         - The preprocessor object is returned which can be used for transforming the data
         - Exception handling is implemented to catch any errors during the transformation process
         - Logging is implemented to log the steps of the transformation process
        '''
        try:
            logging.info('Data Transformation initiated')

            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender',
                                    'race/ethnicity',
                                    'parental level of education',
                                    'lunch', 
                                    'test preparation course']
            

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])  

            logging.info('Numerical and Categorical pipeline created')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])


            return preprocessor
        

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self,train_path,test_path):

        '''
        This function is responsible for initiating the data transformation process
         - It reads the train and test data from the given paths
         - It obtains the preprocessing object by calling the get_data_transformer_object function
         - It separates the input features and target feature from the train and test data
         - It applies the preprocessing object on the input features of the train and test data
        - It concatenates the transformed input features with the target feature to create the final train and test arrays
         '''

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error("Error occurred while initiating data transformation")
            raise CustomException(e,sys)