'''This script takes in a test dataset (in .csv format) and uses the random forest model trained to output scores of m6a modification for each transcript id and position
It assumes the saved model is stored in the same directory as this script, under the name 'randomforest.pkl'
Example usage: python randomforest_predict.py ../data/dataset1.csv'''

# Import dependencies
import pandas as pd
from pathlib import Path
import pickle
import os
import argparse


# Read in test data
parser = argparse.ArgumentParser(description='Predict probability scores for test data.')
parser.add_argument('test_data', type=str, help='Path to the test data file. Can be relative path, relative to the folder containing this script.')
args = parser.parse_args()
data_path = args.test_data
base_path = Path(__file__).parent
file_path = (base_path / data_path).resolve()
data = pd.read_csv(file_path)


# Specify columns to be used for prediction
cols_used = ['transcript_position', 'time_1_mean', 'time_1_median', 'time_1_std',
             'stddev_1_mean', 'stddev_1_median', 'stddev_1_std',
             'current_1_mean', 'current_1_median', 'current_1_std', 'current_1_range',
             'time_2_mean', 'time_2_median', 'time_2_std',
             'stddev_2_mean', 'stddev_2_median', 'stddev_2_std',
             'current_2_mean', 'current_2_median', 'current_2_std', 'current_2_range',
             'time_3_mean', 'time_3_median', 'time_3_std',
             'stddev_3_mean', 'stddev_3_median', 'stddev_3_std', 
             'current_3_mean', 'current_3_median', 'current_3_std', 'current_3_range',
             'transcript_id_encoded', '6-seq_encoded']


# Create a copy of data for frequency encoding
data_encoded = data.copy()

# Frequency encoding for transcript_id
transcript_id_freq = data['transcript_id'].value_counts().to_dict()
data_encoded['transcript_id_encoded'] = data_encoded['transcript_id'].map(transcript_id_freq)

# Frequency encoding for 6-seq
six_seq_freq = data['6-seq'].value_counts().to_dict()
data_encoded['6-seq_encoded'] = data_encoded['6-seq'].map(six_seq_freq)

X_final = data_encoded[cols_used]


# Load model saved in .pkl formate
pkl_path = (base_path / 'randomforest.pkl').resolve()
with open(pkl_path, 'rb') as file:
    rf_model = pickle.load(file)


# Predict for test data
y_predict = rf_model.predict_proba(X_final)
df_dict = {'transcript_id': data['transcript_id'],
      'transcript_position': data['transcript_position'],
      'score': y_predict[:, 1]}
df = pd.DataFrame(df_dict)


# Save predictions
root, extension = os.path.splitext(data_path)
new_filename = os.path.basename(root) + '_predict.csv'   # Output will be saved as original filename, with a '_predict' appended to it
result_directory = base_path / 'result'   # Save output in 'result' directory, within the current directory that this script is in
if not result_directory.exists():
    result_directory.mkdir(parents=True)
save_path = (result_directory / new_filename).resolve()
df.to_csv(save_path, index=False)