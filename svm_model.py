import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path

base_path = Path(__file__).parent
file_path = (base_path / 'test1.csv').resolve() #Add in file name
data = pd.read_csv(file_path)

#Create dataframes to score evalaution metrics
columns = ['linear_0.1', 'linear_1', 'linear_10',
           'rbf_0.1', 'rbf_1', 'rbf_1',
           'string_0.1', 'string_1', 'string_10']
index = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean']

f1score_df = pd.DataFrame(np.nan, columns = columns, index = index)
precision_df = pd.DataFrame(np.nan, columns = columns, index = index)
recall_df = pd.DataFrame(np.nan, columns = columns, index = index)
accuracy_df = pd.DataFrame(np.nan, columns = columns, index = index)


#Conduct 5-fold Cross Validation
kernels = ['linear', 'rbf', 'string'] #parameter
C = [0.1, 1, 10] #parameter

cols_used = ['transcript_position', 'time_1', 'stddev_1', 'mean_current_1',
             'time_2', 'stddev_2', 'mean_current_2'
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
            test = data.loc[data['fold'] == fold, ]
            train = data.loc[data['fold' != fold], ]

            #frequency encoding for transcript id
            transcript_frequencies_test = test['transcript_id'].value_counts().to_dict()
            test['transcript_id_encoded'] = test['transcript_id'].map(transcript_frequencies_test)
            transcript_frequencies_train = train['transcript_id'].value_counts().to_dict()
            train['transcript_id_encoded'] = train['transcript_id'].map(transcript_frequencies_train)

            #frequency encoding for 6-seq
            six_seq_test = test['6-seq'].value_counts().to_dict()
            test['6-seq_encoded'] = test['6-seq'].map(six_seq_test)
            six_seq_train = train['6-seq'].value_counts().to_dict()
            train['6-seq_encoded'] = train['6-seq'].map(six_seq_train)

            X_test = test[cols_used] #use columns 1-12 as inputs
            Y_test = test['label'] #prediction
            X_train = train.iloc[cols_used]
            Y_train = train.iloc['label']
            svm.fit(X_train, Y_train)
            probabilities = svm.predict_proba(X_test)

            #set threshold; if probability > threshold count it as positive
            threshold = 0.5
            binary_pred = (probabilities[:, 1] >= threshold).astype(int)

            #calculate metrics
            f1score = f1_score(Y_test, binary_pred)
            precision = precision_score(Y_test, binary_pred)
            recall = recall_score(Y_test, binary_pred)
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
        
        


