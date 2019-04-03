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

app = Flask(__name__)

# Do data processing here?


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route('/predict', methods=['POST'])
#Function to accept user input and recommened new craft beers - user input to be 3 inputs
def predict():
    try:    
        print("one")
        input_test = pd.DataFrame(request.json) #JSON input from user
        print("two")
        input_test['beer_beerid'] = pd.DataFrame(beer3.loc[beer3['beer_name'].isin(input_test['beer_name']), 'beer_beerid'].unique()) #Obtain beer id for beer name given by user
        print("three")
        input_test['userId'] = input_test['userId'].astype(str) #Convert userId column to appropriate format for append
        print("four")
        frame = beer3.append(input_test, sort=True) #Append info to dataframe of all beer reviews 
        print("five")

        frame[['userId','beer_beerid']] = frame[['userId', 'beer_beerid']].apply(lambda x: x.astype(str)) #Convert columns to appropriate format
        frame['review_overall'] = frame['review_overall'].astype('float64')
        
        iids = frame['beer_beerid'].unique() #Obtain list of all beer Ids
        iids2 = frame.loc[frame['userId'].isin(input_test['userId']), 'beer_beerid'] #Obtain list of ids that user has rated
        iids_to_pred = np.setdiff1d(iids,iids2) #List of all beers user didn't rate

        testtest = [['user', beer_beerid, 4.5] for beer_beerid in iids_to_pred] #Array of beers to predict for users      
        predictions2 = pd.DataFrame(svd.test(testtest)) #Predict and convert to DataFrame
        
        predictions2 = predictions2.sort_values(by=['est'], ascending = False)[:5] #Obtain top 5 predictions
        predictions3 = predictions2.merge(beer3[['beer_name','beer_beerid','beer_abv','beer_style']], left_on='iid',right_on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information

        predictions3 = predictions3[['beer_name','beer_abv','beer_style']].to_json(orient='index') #Convert desired output to json for iOS output

        return jsonify({"prediction": predictions3})
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 5000 # If you don't provide any port then the port will be set to 12345

    # Read beers from csv
    beer3 = pd.read_csv('beerCSV.csv')
    #Convert columns to appropriate format
    beer3[['userId','beer_beerid']] = beer3[['userId', 'beer_beerid']].apply(lambda x: x.astype(str))
    #Create and prepare training set for model input
    reader = Reader(rating_scale=(1, 5))
    training_set = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
    training_set = training_set.build_full_trainset()

    #Set model parameters - kNN & SVD
    sim_options = {
        'name': 'pearson_baseline',
        'user_based': True
    }
 
    knn = KNNBasic(sim_options=sim_options, k=10)
    svd = SVD()

    #Train model
    #knn.fit(training_set)
    svd.fit(training_set)

    #Save Model
    joblib.dump(svd, 'recommender_model')

    #Load Model for API
    svd_iOS = joblib.load('recommender_model')
    print("model loaded")
    print(predict())

    
    app.run(port=port, debug=True)
