import pandas as pd
import numpy as np
import pickle
import os
import json


def load_model(model_path):
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    return model 

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop("corporation", axis=1)
    y = df.exited.values
    x = df.drop("exited", axis=1)
    return (x, y)