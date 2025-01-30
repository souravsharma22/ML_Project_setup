import os
import sys
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.custom_exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Starting model training")
            x_train,y_train,x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            logging.info("Train test split completed")
            models={
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "Catboost":CatBoostRegressor(),
                "Xgboost":XGBRegressor(),
                "LinearRegressor":LinearRegression(),
                "GradientBoost":GradientBoostingRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "K-neighbour":KNeighborsRegressor()
            }
            model_report:dict=evaluate_model(x_test=x_test,y_test=y_test,x_train=x_train,y_train=y_train,models=models)
            logging.info("All models have been evaluated")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model Found")
            logging.info("Best Model found on data")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            logging.info("Model has been saved")

            predicted = best_model.predict(x_test)

            r2_score1 = r2_score(y_pred=predicted,y_true=y_test)
            return r2_score1

        except Exception as e:
            raise CustomException(e,sys)