import training
import scoring
import deployment
import diagnostics
import reporting
import os
import ingestion
import json
import utils

with open('config.json','r') as f:
    config = json.load(f) 


new_data = os.path.join(config["output_folder_path"], "finaldata.csv")
model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
clean_data = utils.load_data(new_data)
old_f1_score = os.path.join(config["prod_deployment_path"], "latestscore.txt")


##################Check and read new data
#first, read ingestedfiles.txt
with open('production_deployment/ingestedfiles.txt', 'r') as f:
    ingestedfiles = f.read()
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
sourcedata_folder = os.listdir('sourcedata')


# Determing whether the source data folder has files that aren't listed in ingestedfiles
new_files_found = False
for file in sourcedata_folder:
    if file not in ingestedfiles:
        print(file)
        new_files_found = True

# If new files found, then ingest it and score model with new data.
if new_files_found:
    ingestion.merge_multiple_dataframe()

    # Calculate new f1 score
    new_f1_score = scoring.score_model(clean_data, model_path)
    print(new_f1_score)

    # reading old f1 score
    with open(old_f1_score, "r") as f:
        old_f1score = float(f.read().replace("F1_score = ", ""))

    ################## Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    # checking new and old F1 scores
    if new_f1_score > old_f1score:
        print(f"NO MODEL DRIFT FOUND: {new_f1_score} > {old_f1score}")
        print("EXITING THE PROGRAMM")
        exit()
    else:
        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        print(f"MODEL DRIFT! {new_f1_score} < {old_f1score}")
        # Retrain and redeploy model
        training.train_model(config)
        deployment.store_model_into_pickle(config)
        
        
        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
        sp.call(["python", "reporting.py"])
        sp.call(["python", "apicalls.py"])












