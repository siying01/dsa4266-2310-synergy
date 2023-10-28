# assumes that the pickle file and the parse json script are in the same folder as this script
# run this script in the command line by:
# python generate_predictions.py path/to/data/folder

import subprocess
import sys
import os
from pathlib import Path


base_path = Path(__file__).parent
data_folder_path = sys.argv[1]


# parse json into csv
data_files = os.listdir(data_folder_path)
parsing_script_path = 'parse_json_generalised.py'    # TO CHANGE IF FILE NAME CHANGES
for data in data_files:
    data_path = (base_path / data_folder_path / data).resolve()
    subprocess.run(['python', parsing_script_path, data_path, 'test'], check=True, shell=True)
    

# predict for each csv data file
# csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
# predict_script_path = 'randomforest_predict.py'    # TO CHANGE IF FILE NAME CHANGES
# for data in csv_files:
#     data_path = (base_path / data_folder_path / data).resolve()
#     subprocess.run(['python', predict_script_path, data_path], check=True, shell=True)