'''This script takes in a data file containing bag-level representations, and trains an Adaboost classfier with it.
For more information, refer to the help page: python adaboost_train.py --help
Example usage: python adaboost_predict.py ../../../data/dataset1.csv'''

# import dependencies
import os
import argparse
import pickle
import pandas as pd
from pathlib import Path


# read in data
parser = argparse.ArgumentParser(description='Predict probability scores for test data.')
parser.add_argument('test_data', type=str, help='Path to the test data file. Can be relative path, relative to the folder containing this script.')
args = parser.parse_args()
data = args.test_data
base_path = Path(__file__).parent
file_path = (base_path / data).resolve()
df = pd.read_csv(file_path)


# compute bag-level representation for the test data
def kurt(x):
    return x.kurt()
agg_funcs = {
    "time_1": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "stddev_1": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "mean_current_1": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "time_2": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "stddev_2": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "mean_current_2": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "time_3": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "stddev_3": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "mean_current_3": ["mean", "std", "sem", "skew", kurt, "max", "min"]
}
result = df.groupby(["transcript_id","transcript_position"], as_index=False).agg(agg_funcs)
result.columns = ['transcript_id', 'transcript_position'] + [f'{col[0]}_{col[1]}' for col in result.columns[2:]]


# restore model from pickle file
pkl_path = (base_path / 'adaboost_classifier.pkl').resolve()
with open(pkl_path, 'rb') as file:
    adaboost_classifier = pickle.load(file)


# output predictions
X = result.loc[:, "time_1_mean":"mean_current_3_min"]
class_probabilities = adaboost_classifier.predict_proba(X)
output_dict = {'transcript_id': result['transcript_id'],
               'transcript_position': result['transcript_position'],
               'score': class_probabilities[:, 1]}
output = pd.DataFrame(output_dict)


# save predictions
root, extension = os.path.splitext(data)
new_filename = os.path.basename(root) + '_predict.csv'
result_directory = base_path / 'result'
if not result_directory.exists():
    result_directory.mkdir(parents=True)
save_path = (result_directory / new_filename).resolve()
output.to_csv(save_path, index=False)