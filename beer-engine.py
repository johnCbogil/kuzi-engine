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

app = Flask(__name__)

# Read beers from csv
beer3 = pd.read_csv('beerCSV.csv')
print("#Read beers from csv")

#Create dataframe of required columns then convert to SFrame for turicreate
beer3_1 = beer3[['userId','beer_beerid','review_overall']]
beer3_1 = tc.SFrame(beer3_1)
beer3_1 = beer3_1.dropna()
print("#Create dataframe of required columns then convert to SFrame for turicreate")

#Create SFrame of additional info on beers for model
beer_info = beer3[['beer_beerid','beer_style','beer_abv']].drop_duplicates()
beer_info = tc.SFrame(beer_info)
print("#Create SFrame of additional info on beers for model")

#Create training and validation set
training_data, validation_data = tc.recommender.util.random_split_by_user(beer3_1, 'userId', 'beer_beerid')
print("#Create training and validation set")

#Create item similarity model
beer_model = tc.item_similarity_recommender.create(training_data, 
                                            user_id="userId", 
                                            item_id="beer_beerid", 
                                            item_data=beer_info,
                                            target="review_overall")
print("#Create item similarity model")           

#Save model
beer_model.save("beer_model")
print("#Save model")

#Load model
beer_model_load = tc.load_model("beer_model")
print("#Load model")

@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route('/predict', methods=['POST'])
#Function to accept user input and recommened new craft beers - user input to be 3 inputs
def predict():
    try:
        input_test = pd.DataFrame(request.json) #JSON input from user
        input_test['beer_beerid'] = pd.DataFrame(beer3.loc[beer3['beer_name'].isin(input_test['beer_name']), 'beer_beerid'].unique()).astype('int64') #Obtain beer id for beer name given by user
        print("#Obtain beer id for beer name given by user")

        predict_frame = tc.SFrame(input_test) #Convert user input dataframe to SFrame
        print("#Convert user input dataframe to SFrame")

        beer_recs = pd.DataFrame(beer_model.recommend(predict_frame['userId'], new_observation_data = predict_frame)) #Predict new beers for user and convert to dataframe
        print("#Predict new beers for user and convert to dataframe")

        beer_recs_final = beer_recs.merge(beer3[['beer_name','beer_beerid','beer_abv','beer_style']], on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information
        print("#Join predictions to beer3 to obtain additional information")

        beer_recs_final = beer_recs_final[['beer_name','beer_abv','beer_style']].to_json(orient='records') #Convert desired output to json for iOS output
        print("#Convert desired output to json for iOS output")
        
        return beer_recs_final    

    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 5000 # If you don't provide any port then the port will be set to 12345

   
    
    app.run(port=port, debug=True)
