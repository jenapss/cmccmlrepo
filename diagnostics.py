
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess as sp
##################Load config.json and get environment variables
import utils 
import sys

##################Function to get model predictions
def model_predictions(model_path, data):
    #read the deployed model and a test dataset, calculate predictions
    X_test, _ = data

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    return preds
    
    #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(data_path):
    #calculate summary statistics here
    df = pd.read_csv(data_path)
    X = df.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    stats = ((col, [X[col].mean(), X[col].median(), X[col].std()]) for col in X)
    return list(stats)  #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    #return a list of 2 timing values in seconds
    proccessing_times = list()
    for file in ['ingestion.py', 'training.py']:
        t0 = timeit.default_timer()
        sp.call(['python3', file])
        proccessing_times.append(timeit.default_timer()-t0)
    
    print(proccessing_times,1)
    return proccessing_times

def na_count(data_path):
    data = pd.read_csv(os.path.join(os.getcwd(),data_path,'finaldata.csv'))
    na_percent = list(data.isna().sum(axis=1)/data.shape[0])
    print(na_percent)
    return na_percent

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    args = [sys.executable, "-m", "pip", "list", "--outdated"]
    results = sp.run(args, capture_output=True, check=True).stdout
    indented_results = ("\n" + results.decode()).replace("\n", "\n    ")
    print('+++++++')
    return indented_results
    

if __name__ == '__main__':
    with open('config.json','r') as f:
        config = json.load(f)
    
    model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
    data_tuple = utils.load_data(
    data_path=os.path.join(config["test_data_path"], "testdata.csv")
    )
    model_predictions(os.path.join(config["prod_deployment_path"], "trainedmodel.pkl"),data_tuple)
    dataframe_summary(os.path.join(config["output_folder_path"], "finaldata.csv"))
    execution_time()
    na_count(config["output_folder_path"])
    outdated_packages_list()





    
