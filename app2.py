import uvicorn
from fastapi import FastAPI, HTTPException, Query
from typing import List
import numpy as np
import pickle
import json
import os
import diagnostics as dgns
import scoring
import utils

app = FastAPI()

app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prediction_model = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
score_path = config["prod_deployment_path"]


@app.get("/prediction")
def predict(data_path: str):
    X_y_data = utils.load_data(data_path)
    preds = dgns.model_predictions(prediction_model, X_y_data)
    return {"predictions": list(preds.tolist())}



@app.get("/scoring")
def scoring_model(data_path: str):
    data_frame_tuple = utils.load_data(data_path)
    f1_score = scoring.score_model(data_frame_tuple, prediction_model)
    return {"f1_score": f1_score}


@app.get("/summarystats")
def summary_stats(data_path: str):
    res = dgns.dataframe_summary(data_path)
    return {"summary_statistics": res}


@app.get("/diagnostics")
def diagnose(data_path: str):
    processing_times = dgns.execution_time()
    missing_data = dgns.na_count(config["output_folder_path"])
    outdated_pck = dgns.outdated_packages_list()
    return {
        "execution_time": processing_times,
        "missing_data": missing_data,
        "outdated_packages": outdated_pck
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
