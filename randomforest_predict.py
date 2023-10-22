import pandas as pd
from pathlib import Path
import pickle

base_path = Path(__file__).parent
file_path = (base_path / 'dataset1.csv').resolve() #Add file name
data = pd.read_csv(file_path)

#Columns to be used for prediction
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

#create a copy of data for encoding
data_encoded = data.copy()

#frequency encoding for transcript id
transcript_id_freq = data['transcript_id'].value_counts().to_dict()
data_encoded['transcript_id_encoded'] = data_encoded['transcript_id'].map(transcript_id_freq)

#frequency encoding for 6-seq
six_seq_freq = data['6-seq'].value_counts().to_dict()
data_encoded['6-seq_encoded'] = data_encoded['6-seq'].map(six_seq_freq)

X_final = data_encoded[cols_used]

#load pickle file
pkl_path = (base_path / 'randomforest.pkl').resolve()
with open(pkl_path, 'rb') as file:
    rf_model = pickle.load(file)

y_predict = rf_model.predict_proba(X_final)

df_dict = {'transcript_id': data['transcript_id'],
      'transcript_position': data['transcript_position'],
      'score': y_predict[:, 1]}

df = pd.DataFrame(df_dict)

save_path = (base_path / 'dataset1_predict.csv').resolve()
df.to_csv(save_path, index = False)