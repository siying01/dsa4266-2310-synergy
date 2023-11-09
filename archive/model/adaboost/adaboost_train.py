'''This script takes in a data file and trains an Adaboost classfier with it.
For more information, refer to the help page: python adaboost_train.py --help
Example usage: python adaboost_train.py ../../../data/dataset0.csv'''

# import dependencies
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


# read in data
parser = argparse.ArgumentParser(description='Trains an Adaboost classifier.')
parser.add_argument('training_data', type=str, help='Path to the data file. Can be relative path, relative to the folder containing this script.')
args = parser.parse_args()
data = args.training_data
base_path = Path(__file__).parent
file_path = (base_path / data).resolve()
df = pd.read_csv(file_path)


# compute bag-level representations
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
    "mean_current_3": ["mean", "std", "sem", "skew", kurt, "max", "min"],
    "label": ["mean"]
}
result = df.groupby(["gene_id", "transcript_id","transcript_position"], as_index=False).agg(agg_funcs)
result.columns = [f'{col[0]}_{col[1]}' for col in result.columns]
result = result.rename(columns={"label_mean": "label"})
result['label'] = result['label'].astype(int)


# initialise model
df = result
X = df.loc[:, "time_1_mean":"mean_current_3_min"]
y = df.loc[:, "label"]
minority_perc = sum(y==1) / len(y)
sample_weights = [1.0 if label==0 else 1/minority_perc for label in y]  
adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                                        n_estimators=100,
                                        learning_rate=0.1)


# fit model
adaboost_classifier.fit(X, y, sample_weight=sample_weights)


# save model as pickle file
pkl_path = (base_path / 'adaboost_classifier.pkl').resolve()
with open(pkl_path, 'wb') as file:
    pickle.dump(adaboost_classifier, file)