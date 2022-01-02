from flask import Flask, request, jsonify, Response
import os
import json

from keras.models import load_model

import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

import glob
# import pyarrow.parquet as pa
# from dateutil import parser
# import datetime

# from sklearn.preprocessing import OneHotEncoder , normalize
# import numpy as np
# from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder

# import joblib
# from functools import partial
# from peewee import *
# import peewee as pe
# import pandas as pd
# import xgboost as xgb

# Init app

app= Flask(__name__)


# Load ML model and predict the activity
@app.route('/ml', methods=['GET','POST','PUT'])
def ml_model():
    req_data=request.get_json()

    y=pd.read_json(req_data)
    req_data= np.array(y).reshape(y.shape[0], 1, y.shape[1])
    loaded_model = load_model('LSTM.h5')
    prediction=loaded_model.predict(req_data)
    prediction = np.argmax(prediction, axis=1)
    return json.dumps(str(prediction))


# Run Server
if __name__=='__main__':

    # app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)