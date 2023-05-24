from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics as dgns
import scoring
import utils

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prediction_model = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
score_path = config["prod_deployment_path"]




#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS', "GET"])
def predict():        
    #call the prediction function you created in Step 3
    data_path = request.args.get('data_path')
    X_y_data = utils.load_data(data_path)
    preds = dgns.model_predictions(prediction_model, X_y_data)
    return str(preds) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_model():        
    #check the score of the deployed model
    data_path = request.args.get('data_path')
    data_frame_tuple = utils.load_data(data_path)
    f1_score = scoring.score_model(data_frame_tuple, prediction_model)
    return "SAVED F1 SCORE of {}".format(f1_score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():  
    data_path = request.args.get('data_path')
    res = dgns.dataframe_summary(data_path)

    #check means, medians, and modes for each column
    return str(res) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    data_path = request.args.get('data_path')
    processing_times = dgns.execution_time()
    missing_data = dgns.na_count(config["output_folder_path"])
    outdated_pck = dgns.outdated_packages_list()
    return str(f'Execution time: {processing_times} \n Missing data(%): {missing_data} \n Outdated packages: {outdated_pck}')

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
