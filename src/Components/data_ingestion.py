import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass                      # this class has only to define variable hence we use dataclass
class DataIngestion_config:              
    train_data_path:str = os.path.join('artifacts','train.csv')     # data ingestion component saves file in these paths
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):                                     # INITIALIZE DataIngestion class
        self.ingestion_config = DataIngestion_config()      #  initialize object and adding in a variable

    def initiate_data_ingestion(self):
        logging.info("Entered in data ingestion method")
        try:
            df = pd.read_csv('notebook\data\stud.csv')       # READING CSV FILE FROM NOTEBOOK
            logging.info('reading dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok =True)  # IF DIRECTORY NOT AVAILABLE CREATE OTHERWISE USE EXISTING
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)               # convert dataframe to csv to raw data path
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)              # do train test split
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)      # convert train_set to csv
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info("Ingestion of data component")

            return (
                self.ingestion_config.train_data_path,                                          
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    #obj.initiate_data_ingestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    #data_transformation.initiate_data_transformation(train_data,test_data)
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))




