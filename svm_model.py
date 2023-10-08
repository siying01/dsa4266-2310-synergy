import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path

base_path = Path(__file__).parent
file_path = (base_path / 'test1.csv').resolve() #Add in file name
data = pd.read_csv(file_path)

#print(data.dtypes)

#Create dataframes to score evalaution metrics
columns = ['linear_0.1', 'linear_1', 'linear_10',
           'rbf_0.1', 'rbf_1', 'rbf_10',
           'poly_0.1', 'poly_1', 'poly_10']
index = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean']

f1score_df = pd.DataFrame(np.nan, columns = columns, index = index)
precision_df = pd.DataFrame(np.nan, columns = columns, index = index)
recall_df = pd.DataFrame(np.nan, columns = columns, index = index)
accuracy_df = pd.DataFrame(np.nan, columns = columns, index = index)


#Conduct 5-fold Cross Validation
kernels = ['linear', 'rbf', 'poly'] #parameter
C = [0.1, 1, 10] #parameter

cols_used = ['transcript_position', 'time_1', 'stddev_1', 'mean_current_1',
             'time_2', 'stddev_2', 'mean_current_2', 
             'time_3', 'stddev_3', 'mean_current_3', 
             'transcript_id_encoded', '6-seq_encoded']

count_col = 0
for kernel in kernels:
    for val in C:
        svm = SVC(kernel = kernel, C = val, 
                  probability = True, class_weight = 'balanced') #class_weight = 'balanced' gives more importance to minority class
        #do 5-fold CV
        count_row = 0 #keep track of current row
        for fold in range(1, 6):
            #separate into test and training
            test = data.loc[data['folds'] == fold, ]
            train = data.loc[data['folds'] != fold, ]

            #copy of df
            test_freq = test.copy()
            train_freq = train.copy()

            #frequency encoding for transcript id
            transcript_id_test = test['transcript_id'].value_counts().to_dict()
            test_freq['transcript_id_encoded'] = test_freq['transcript_id'].map(transcript_id_test)
            transcript_id_train = train['transcript_id'].value_counts().to_dict()
            train_freq['transcript_id_encoded'] = train_freq['transcript_id'].map(transcript_id_train)

            #frequency encoding for 6-seq
            six_seq_test = test['6-seq'].value_counts().to_dict()
            test_freq['6-seq_encoded'] = test_freq['6-seq'].map(six_seq_test)
            six_seq_train = train['6-seq'].value_counts().to_dict()
            train_freq['6-seq_encoded'] = train_freq['6-seq'].map(six_seq_train)

            X_test = test_freq[cols_used] #select columns to be used as variables for training
            Y_test = test_freq['label'] #prediction
            X_train = train_freq[cols_used]
            Y_train = train_freq['label']
            svm.fit(X_train, Y_train)
            probabilities = svm.predict_proba(X_test)

            #set threshold; if probability > threshold count it as positive
            threshold = 0.5
            binary_pred = (probabilities[:, 1] >= threshold).astype(int)

            #calculate metrics
            f1score = f1_score(Y_test, binary_pred, zero_division = 0.0) #check zero_division and explore other metrics
            precision = precision_score(Y_test, binary_pred, zero_division = 0.0)
            recall = recall_score(Y_test, binary_pred, zero_division = 0.0)
            accuracy = accuracy_score(Y_test, binary_pred)

            #update dataframe with metric scores
            f1score_df.iloc[count_row, count_col] = f1score
            precision_df.iloc[count_row, count_col] = precision
            recall_df.iloc[count_row, count_col] = recall
            accuracy_df.iloc[count_row, count_col] = accuracy
        

            count_row += 1
            if count_row == 5:
                f1score_df.iloc[count_row, count_col] = f1score_df.iloc[:5, count_col].mean() #obtain mean of f1 scores
                precision_df.iloc[count_row, count_col] = precision_df.iloc[:5, count_col].mean() #obtain mean of precision scores
                recall_df.iloc[count_row, count_col] = recall_df.iloc[:5, count_col].mean() #obtain mean of recall scores
                accuracy_df.iloc[count_row, count_col] = accuracy_df.iloc[:5, count_col].mean() #obtain mean of accuracy scores
        count_col += 1

#print(f1score_df)
#print(precision_df)
#print(recall_df)
#print(accuracy_df)
        
        


