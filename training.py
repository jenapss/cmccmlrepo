from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from pandas.core.construction import array
from typing import Tuple
import utils

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


   

#################Function for training the model
def train_model(config):
    csv_dataset = os.path.join(config['output_folder_path'], 'finaldata.csv')
    
    X, y = utils.load_data(csv_dataset)
    
    #use this logistic regression for training
    log_reg_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    print('Started training model')
    log_reg_model.fit(X,y)
    print('Finished training model')
    #write the trained model to your workspace in a file called trainedmodel.pkl

    if not os.path.exists(config['output_model_path']):
        os.makedirs(config['output_model_path'])

    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    pickle.dump(log_reg_model, open(model_path, 'wb'))
    print('Saved the trained model to a file called trainedmodel.pkl')


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    #data = load_data(config)
    #print(type(data[0]))
    #print(type(data[1]))
    train_model(config)
    #print(type(data))