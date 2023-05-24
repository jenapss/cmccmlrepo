from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


####################function for deployment
def store_model_into_pickle(config):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    trained_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    latest_score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    ingestedfiles_path = os.path.join(config['output_folder_path'], 'ingestedfiles.txt')    

    path_list = [trained_model_path, latest_score_path, ingestedfiles_path]

    if not os.path.exists(os.path.join(config['prod_deployment_path'])):
        os.makedirs(os.path.join(config['prod_deployment_path']))

    for path in path_list:
        shutil.copy(src=path, dst=os.path.join(config['prod_deployment_path']))
        print('COPIED {}'.format(path))


if __name__ ==  '__main__':
    with open('config.json','r') as f:
        config = json.load(f)
    store_model_into_pickle(config)

