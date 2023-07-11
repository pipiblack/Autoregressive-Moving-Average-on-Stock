#library imports
# import pydantic
from pydantic import BaseModel
from fastapi import FastAPI

import joblib

import pandas as pd
import numpy as np
import json

# create instance of the Flask class

app = FastAPI()

#create a class to imort the model
class model_import(BaseModel):
    Quantity: int
    UnitPrice: float
    Country: str
    Date : str
    Hour : int
    Minute : int

# load the saved model
model = joblib.load('arima_model.pkl')

@app.post('/predict_sales')
# create a function to predict the sales

def predict_sales(input_params: model_import):
    # make prediction
    input_data = pd.DataFrame([input_params.json()])
    input_dictionary = input_data.to_dict('records')[0]

    # convert the date to dictionary format
    Quantity = input_dictionary['Quantity']
    UnitPrice = input_dictionary['UnitPrice']
    Country = input_dictionary['Country']
    Date = input_dictionary['Date']
    Hour = input_dictionary['Hour']
    Minute = input_dictionary['Minute']

    input_list  = [Quantity, UnitPrice, Country, Date, Hour, Minute]
    input_array = np.array(input_list).reshape(1, -1)

    # prediction of list values

    prediction = model.predict(input_list)

    # return the prediction
    return {
        'prediction': prediction
    }





