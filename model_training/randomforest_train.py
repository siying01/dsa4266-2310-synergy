'''This script takes in the parsed csv training file with labels as input to train the random forest model
Example usage: python randomforest_train.py ../data/dataset0.csv'''

# Import dependencies
import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle

base_path = Path(__file__).parent
data_folder_path = sys.argv[1]
file_path_train = (base_path / data_folder_path).resolve()
train_data = pd.read_csv(file_path_train)


# Specify columns to be used for training and prediction
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
train_data_encoded = train_data.copy()

# Frequency encoding for transcript_id
transcript_id_train = train_data['transcript_id'].value_counts().to_dict()
train_data_encoded['transcript_id_encoded'] = train_data_encoded['transcript_id'].map(transcript_id_train)

# Frequency encoding for 6-seq
six_seq_train = train_data['6-seq'].value_counts().to_dict()
train_data_encoded['6-seq_encoded'] = train_data_encoded['6-seq'].map(six_seq_train)

X_final = train_data_encoded[cols_used]
Y_final = train_data_encoded['label']


# Oversample training data
oversampler = RandomOverSampler(sampling_strategy = 0.2)
X_final, Y_final = oversampler.fit_resample(X_final, Y_final)


# Initialise random forest model
rf_model = RandomForestClassifier(n_estimators = 200, max_depth = 70,
                                  class_weight = 'balanced_subsample', n_jobs = -1)


# Fit model
rf_model.fit(X_final, Y_final)


# Save model as .pkl file
pkl_path = (base_path / 'randomforest.pkl').resolve()
with open(pkl_path, 'wb') as file:
    pickle.dump(rf_model, file)