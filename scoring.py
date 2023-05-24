from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score
import utils


#################Function for model scoring
def score_model(training_data, model_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    X, y = training_data

    log_reg_model = utils.load_model(model_path)
    preds = log_reg_model.predict(X)

    with open("config.json", "r") as f:
        config = json.load(f)

    # compute model F1 score
    score_result = f1_score(y, preds)
    if not os.path.exists(config["output_model_path"]):
        os.makedirs(config["output_model_path"])
    with open(os.path.join(config["output_model_path"], "latestscore.txt"), "w") as score:
        score.write(f"F1_score = {str(score_result)}")
    return score_result


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
    path_to_model = os.path.join(config["output_model_path"], "trainedmodel.pkl")
    path_to_score = config["output_model_path"]
    df_tuple = utils.load_data(test_data_path)
    score_model(df_tuple, path_to_model)
    