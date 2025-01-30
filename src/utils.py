import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.custom_exception import CustomException

def save_object (file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            param = params[list(models.keys())[i]]
            # model.fit(x_train,y_train)
            gd = GridSearchCV(estimator=model,param_grid=param,cv=3)

            gd.fit(x_train,y_train)

            model.set_params(**gd.best_params_)
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_true=y_train,y_pred=y_train_pred)
            test_model_score = r2_score(y_true=y_test,y_pred=y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
