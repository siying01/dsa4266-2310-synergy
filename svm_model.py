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
            X_test = test.iloc[:, 1:13] #use columns 1-12 as inputs
            Y_test = test.iloc[:, 14] #prediction
            X_train = train.iloc[:, 1:13]
            Y_train = train.iloc[:, 14]
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
        count_col += 1
        
        


