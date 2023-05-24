import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import plot_confusion_matrix
import utils
import diagnostics as dgn
from joblib import load

with open('config.json','r') as f:
        config = json.load(f) 
data_path = os.path.join(config["test_data_path"], "testdata.csv")
model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
plot_path=config["output_model_path"]


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    model = load(model_path)
    X, y = utils.load_data(data_path)

    plot_confusion_matrix(model, X,y)
    plt.savefig(os.path.join(plot_path, "conf_matrix.png"))



if __name__ == '__main__':
    ###############Load config.json and get path variables
    
    score_model()
