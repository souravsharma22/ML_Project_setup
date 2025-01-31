import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import Customdata

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/prediction",methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=Customdata(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        input_df = data.get_data_as_data_frame()
        print(input_df)

        predict_pipelin=PredictPipeline()
        results=predict_pipelin.predict(input_df) # as the output will be in lit format
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")