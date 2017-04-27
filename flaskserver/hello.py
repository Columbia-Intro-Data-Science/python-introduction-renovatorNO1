import sys
import os
import shutil
import time
import traceback
import numpy as np

from flask import Flask, request, jsonify, send_from_directory, render_template

import pandas as pd
from sklearn.externals import joblib


app = Flask(__name__)

training_data = 'data/brooklyn_clean_final.csv'

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory


@app.route('/', methods=['GET', 'POST'])
def login():
    return render_template('form.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    import json
    if request.method == 'POST' or request.method == 'GET':   
      result = request.form
      #print (json.dumps(result,  default=lambda o: o.__dict__))
      json_string = json.dumps(result,  default=lambda o: o.__dict__)
      json_ = json.loads(json_string)
      for key, value in json_.items():
          json_[key] = int(value)
          #for key, value in pair.iteritems():
              #print(key, value)
      return render_template("result.html",result = result)    
'''    
@app.route("/")
def hello():
    return "Hello World!"
'''
@app.route('/predict', methods=['POST'])
def predict():
    import json
    if clf:
        try:
            #json_ = request.json
            result = request.form
            #print (json.dumps(result,  default=lambda o: o.__dict__))
            json_string = json.dumps(result,  default=lambda o: o.__dict__)
            json_ = json.loads(json_string)
            for key, value in json_.items():
                json_[key] = int(value)
            #query = pd.DataFrame(json_)
            
            
            #data = {'RESIDENTIAL_UNITS':2, 'COMMERCIAL_UNITS':1, 'GROSS_SQUARE_FEET':4824, 'AGE':107,'11207':0, '11239':0,'11236':0,'11208':0,'11234':0,'11203':0,'11212':0,'11224':0,'11210':0,'11229':0,'11233':0,'11228':0,'11204':0,'11214':0,'11221':0,'11209':0,'11235':0,'11213':0,'11223':0,'11220':0,
            # '11219':0,'11218':0,'11230':0,'11226':0,'11232':0,'11237':0,'11216':0,'11225':0,'11231':0,'11215':0,'11222':0,'11217':0,'11238':0,'11206':0,'11205':0,'11211':0,'11249':0,'11201':1,'TAX:1':1,'TAX:2':0,'TAX:4':0}
             
            df_user=pd.DataFrame(data=json_, index=np.arange(1))

            prediction = list(clf.predict(df_user))

            return jsonify({'prediction': prediction})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print ('train first')
        return 'no model here'
    
@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.cross_validation import KFold
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.neighbors import KNeighborsClassifier as KNN
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.utils import shuffle
    from sklearn.metrics import roc_curve, auc
    import pylab
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestClassifier
    import re
    import pylab as plt
    import seaborn
    from sklearn.linear_model import LinearRegression
    import numpy.random as nprnd
    from sklearn.tree import DecisionTreeRegressor

    df = pd.read_csv(training_data, index_col=0)

    X = df.drop(['SALE_PRICE'], 1)
    
    y = df[['SALE_PRICE']]
    y_classified = y.copy(True)
    row_num = X.shape[0]

    def add_zipcode(zipcodes):
        for zipcode in zipcodes:
            zeros = np.zeros(row_num+1)
            s = pd.Series(zeros)
            key = str(zipcode)
            X[key] = s
            for pos in range(0,row_num):
                if X.iloc[pos]['ZIPCODE'] == zipcode:
                    X.set_value(pos+1, key, 1)
                
    def add_taxclass(taxclass):
        for tax in taxclass:
            zeros = np.zeros(row_num+1)
            s = pd.Series(zeros)
            key = str(tax)
            X["TAX:"+key] = s
            for pos in range(0,row_num):
                if X.iloc[pos]['TAX_CLASS'] == tax:
                    X.set_value(pos+1, "TAX:"+key, 1)
                

    def add_buildingclass(buildingclass):
        for building in buildingclass:
            zeros = np.zeros(row_num+1)
            s = pd.Series(zeros)
            key = str(building)
            X["building:"+key] = s
            for pos in range(0,row_num):
                if X.iloc[pos]['TAX_CLASS'] == building:
                    X.set_value(pos+1, "building:"+key, 1)

    zipcodes = [11239, 11236,11208,11207,11234,11203,11212,11224,11210,
            11229,11233,11228,11204,11214,11221,11209,11235,11213,11223,11220,
            11219,11218,11230,11226,11232,11237,11216,11225,11231,11215,11222,11217,11238,11206,11205,11211,11249,11201]
    taxclass = [1,2,4]

    add_zipcode(zipcodes)
    add_taxclass(taxclass)
    #add_buildingclass(buildingclass)

    
    
    # Set the upper bound of the price
    def classify_y(price):
        for pos in range(0,8497):
            if y.iloc[pos]['SALE_PRICE'] >= price:
                y_classified.set_value(pos+1, 'SALE_PRICE', 1)
            else: 
                y_classified.set_value(pos+1, 'SALE_PRICE',0)
            
    # Set the lower bound of the price            
    def classify_y_down(price):
        for pos in range(0,8497):
            if y.iloc[pos]['SALE_PRICE'] <= price:
                y_classified.set_value(pos+1, 'SALE_PRICE', 1)
            else: 
                y_classified.set_value(pos+1, 'SALE_PRICE',0)
            
            
    def classify_y_range(low, high):
        for pos in range(0,8497):
            if y.iloc[pos]['SALE_PRICE'] <= high & y.iloc[pos]['SALE_PRICE'] >= low:
                y_classified.set_value(pos+1, 'SALE_PRICE', 1)
            else: 
                y_classified.set_value(pos+1, 'SALE_PRICE',0)
            
       
    X = X.drop(['ZIPCODE','TAX_CLASS'], 1)
    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(X.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = DecisionTreeRegressor(max_depth=3)
    start = time.time()
    clf.fit(X, y)
    print ('Trained in %.1f seconds' % (time.time() - start))
    print ('Model training score: %s' % clf.score(X, y))

    joblib.dump(clf, model_file_name)

    return 'Success'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80
        
    #app.run(port=port, debug=True)
    

    try:
        clf = joblib.load(model_file_name)
        print ('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print ('model columns loaded')

    except Exception as e:
        print ('No model here')
        print ('Train first')
        print (str(e))
        clf = None

    app.run(port=port, debug=True)
