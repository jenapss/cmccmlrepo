
import json
import os
import requests


def save_responses(URL,saving_path):
   
    response1 = requests.post(URL + "/prediction?data_path=testdata/testdata.csv").text
    response2 = requests.get(URL  + "/scoring?data_path=testdata/testdata.csv").text
    response3 = requests.get(URL  + "/summarystats?data_path=testdata/testdata.csv").text
    response4 = requests.get(URL  + "/diagnostics?data_path=testdata/testdata.csv").text
    responses = str(response1+ '\n' + response2 + '\n'+ response3 + '\n' + response4)

    with open(saving_path, "w") as f:
        f.write(str(responses))
    print("DIAGNOSTICS: OK")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    # set localhost url
    URL = "http://10.8.0.22:8000"
    requests_res_save_path = os.path.join(config["output_model_path"], "apireturns.txt")
    
    res = save_responses(URL,saving_path=requests_res_save_path )
