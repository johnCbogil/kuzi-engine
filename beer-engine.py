import pandas as pd
import numpy as np
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from flask import request
from flask import jsonify
from surprise import KNNBasic, SVD, Reader, Dataset
from surprise.model_selection import cross_validate
import collections
import traceback
import turicreate as tc
from flask import send_file
from pandas.io.json import json_normalize
import json


app = Flask(__name__)

# Read beers from csv
beer2 = pd.read_csv('beer2.csv')
#Load model
beer_model = tc.load_model("beer_model")
print("#Load and read beers from csv")

@app.route("/")
def hello():
    return send_file("tc.jpg", mimetype='image/jpg')
    # return "Welcome to machine learning model APIs!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        jsonDump = json.dumps(request.json)
        input_test = pd.DataFrame(pd.read_json(jsonDump)) #JSON input from user
    
        input_test['beer_beerid'] = pd.DataFrame(beer2.loc[beer2['beer_name'].isin(input_test['beer_name']), 'beer_beerid'].unique()).astype('int64') #Obtain beer id for beer name given by user
        
        predict_frame = tc.SFrame(input_test) #Convert user input dataframe to SFrame
        beer_recs = pd.DataFrame(beer_model.recommend(predict_frame['userId'], new_observation_data = predict_frame)) #Predict new beers for user and convert to dataframe
        
        beer_recs_final = beer_recs.merge(beer2[['beer_name','beer_beerid','beer_abv','beer_style']], on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information
        beer_recs_final = beer_recs_final[['beer_name','beer_abv','beer_style']].to_json(orient='records') #Convert desired output to json for iOS output
        
        return beer_recs_final  

    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 5000 # If you don't provide any port then the port will be set to 12345

   
    
    app.run(port=port, debug=True)
