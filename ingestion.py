import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

############## Check output folder existance - create if not exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    files = os.listdir(f'{os.getcwd()}/{input_folder_path}')
    file_paths = [f"{os.getcwd()}/{input_folder_path}/{i}" for i in files]
    data_frame_cols = pd.read_csv(file_paths[0], nrows=1).columns
    df = pd.DataFrame(columns=data_frame_cols)

    for file in file_paths:
        try:
            df = df.append(pd.read_csv(file)).reset_index(drop=True)
        except:
            print('COULDNT READ AND APPEND {}'.format(file))
    df = df.drop_duplicates()
    df.to_csv(f"{os.getcwd()}/{output_folder_path}/finaldata.csv", index=False)
    
    with open(f"{os.getcwd()}/{output_folder_path}/ingestedfiles.txt", "w") as f:
        f.write(",".join(files))

    print('SAVED finaldata.csv and ingestedfiles.txt into {}'.format(output_folder_path))

if __name__ == '__main__':
    merge_multiple_dataframe()
