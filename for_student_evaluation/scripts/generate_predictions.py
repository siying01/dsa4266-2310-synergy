'''This script takes in the folder containing all the json datasets which the user wish to generate predictions for, and output the predictions
Example usage: python generate_predictions.py ../data'''

# Import dependencies
import subprocess
import sys
import os
from pathlib import Path

base_path = Path(__file__).parent
data_folder_path = sys.argv[1]


# Parse json data into csv
json_files = [f for f in os.listdir(data_folder_path) if f.endswith('.json')]
parsing_script_path = 'parse_json.py' 
for data in json_files:
    data_path = (base_path / data_folder_path / data).resolve()
    result = subprocess.run(['python3', parsing_script_path, data_path, 'test'], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    

# Predict for each csv data file
csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
predict_script_path = 'randomforest_predict.py'
for data in csv_files:
    data_path = (base_path / data_folder_path / data).resolve()
    result = subprocess.run(['python3', predict_script_path, data_path], stdout=subprocess.PIPE, text=True)
    print(result.stdout)