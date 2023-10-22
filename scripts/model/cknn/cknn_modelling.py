'''Set Fold number'''
# choose from 0 to 4, fold = i+1
i = 0

'''Import libraries'''

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (pairwise_distances, balanced_accuracy_score,
                             roc_curve, precision_recall_curve, auc, f1_score,
                             recall_score, precision_score, confusion_matrix)

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

base_path = Path(__file__).parent
file_path = (base_path / "training_data.csv").resolve()

file2_path = (base_path / f"bag_distances_max_validation_fold_{i+1}.npy").resolve()
bag_distances_max = np.load(file2_path)

'''Read data, select columns, create bag id column (transcript_id_position)'''

dataset = pd.read_csv(file_path)

dataset = dataset[['transcript_id', 'transcript_position', '6-seq',
           'time_1', 'stddev_1', 'mean_current_1',
           'time_2', 'stddev_2', 'mean_current_2',
           'time_3', 'stddev_3', 'mean_current_3',
           'gene_id', 'label', 'folds']]

dataset['transcript_id_position'] = dataset['transcript_id'].astype(str) + '-' \
                             + dataset['transcript_position'].astype(str)

dat = dataset

'''Performance metrics'''

def get_roc_auc(y_true, y_pred):
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc

'''Models and Predictions'''

def get_knn_score(train_labels, test_labels, all_bag_distances, n_neighbors):
  # estimated probability = positive class votes / total number of votes
  votes = train_labels[np.argsort(all_bag_distances, axis=1)[:, :n_neighbors]]
  return np.sum(votes, axis = 1) / n_neighbors

def get_citation_counts(train_labels, test_labels, all_bag_distances, n_neighbors):
  test_pos_neg_counts = np.zeros((len(test_labels), 2))
  nn = np.argsort(all_bag_distances.transpose(), axis=1)[:, :n_neighbors]
  for citer in range(len(nn)):
    citations = nn[citer]
    citer_label = np.array([0, 1]) if train_labels[citer] == 1 else np.array([1, 0])
    for citation in citations:
      test_pos_neg_counts[citation, :] += citer_label
  return test_pos_neg_counts

def get_citation_knn_score(train_labels, test_labels, all_bag_distances, n_ref_neighbors=10, n_cit_neighbors=10):
  reference_nn = train_labels[np.argsort(all_bag_distances, axis=1)[:, :n_ref_neighbors]]
  num_pos_neighbors = np.sum(reference_nn, axis=1)
  num_neg_neighbors = n_ref_neighbors - num_pos_neighbors
  reference_nn_labels = np.concatenate([num_neg_neighbors.reshape(-1, 1),
                                        num_pos_neighbors.reshape(-1, 1)], axis=1)
  citer_nn_labels = get_citation_counts(train_labels, test_labels, all_bag_distances, n_cit_neighbors)
  class_counts = reference_nn_labels + citer_nn_labels
  return [bag[1]/np.sum(bag) for bag in class_counts]

def get_citation_knn_score_normalized(train_labels, test_labels, all_bag_distances, n_ref_neighbors=10, n_cit_neighbors=10):
  reference_nn = train_labels[np.argsort(all_bag_distances, axis=1)[:, :n_ref_neighbors]]
  num_pos_neighbors = np.sum(reference_nn, axis=1)
  num_neg_neighbors = n_ref_neighbors - num_pos_neighbors
  reference_nn_labels = np.concatenate([num_neg_neighbors.reshape(-1, 1),
                                        num_pos_neighbors.reshape(-1, 1)], axis=1)
  citer_nn_labels = get_citation_counts(train_labels, test_labels, all_bag_distances, n_cit_neighbors)
  class_counts = reference_nn_labels / np.sum(reference_nn_labels, axis=0) + citer_nn_labels / np.sum(citer_nn_labels, axis=0)
  return [bag[1]/np.sum(bag) for bag in class_counts]

'''Preparation for training'''

# Create dictionaries for numeric coding of distance and model
# results are stored in numpy array before conversion to df
# only numeric in the numpy array

dict_distance = {'max': 0, 'avg': 1, 'min': 2}
dict_model = {'kNN': 4, 'ckNN': 5, 'norm-ckNN': 6}

# set probability threshold for class prediction
knn_threshold_value = 0.5
cknn_threshold_value = 0.5
ncknn_threshold_value = 0.5

# store results
res = []

# store imbalance per fold
train_fold_labels = {}

'''Cross validation'''

print(f"cross validation run: {i+1}")

'''Prepare data'''

# get training and holdout set
dat_train = dat[dat['folds'] != i+1]
dat_test = dat[dat['folds'] == i+1]

# downsample 'negative' 'instances'
dat_train_min = dat_train[dat_train['label'] == 1]
dat_train_maj = dat_train[dat_train['label'] == 0]
dat_train_maj_agg = dat_train_maj.groupby(['transcript_id_position']).agg({'time_1': 'mean', 
                                                        'stddev_1': 'mean', 
                                                        'mean_current_1': 'mean', 
                                                        'time_2': 'mean', 
                                                        'stddev_2': 'mean', 
                                                        'mean_current_2': 'mean',
                                                        'time_3': 'mean', 
                                                        'stddev_3': 'mean', 
                                                        'mean_current_3': 'mean',
                                                        'label': 'first', 
                                                        'folds': 'first'})
dat_train = pd.concat([dat_train_maj_agg, dat_train_min], axis = 0)

### try to reduce RAM here
del dat_train_min
del dat_train_maj
del dat_train_maj_agg

# Check: class imbalance in training set
train_fold_labels[i+1] = dat_train.label.value_counts().to_dict()

# dat_train-or-test: df of 9 features, labels, transcript id and position, 6-seq, gene id
dat_train = dat_train.drop(['folds'], axis = 1)
dat_test = dat_test.drop(['folds'], axis = 1)

# reset index to make sure indexes pair with number of rows
dat_train = dat_train.reset_index(drop = True)
dat_test = dat_test.reset_index(drop = True)

# df_train-or-test: df of 9 features and labels
df_train = dat_train.drop(['transcript_id', 'transcript_position', '6-seq',
                         'gene_id', 'transcript_id_position'], axis = 1)
df_test = dat_test.drop(['transcript_id', 'transcript_position', '6-seq',
                       'gene_id', 'transcript_id_position'], axis = 1)

# arr_train-or-test: np array of 9 features
arr_train = df_train.drop(['label'], axis = 1).values
arr_test = df_test.drop(['label'], axis = 1).values

# X_train-or-test is same as arr_train-or-test but scaled
# no compression of 9 features
scaler = StandardScaler().fit(arr_train)
X_train = scaler.transform(arr_train)
X_test = scaler.transform(arr_test)

## get training bag instances, bag labels
## and bags corresponding to transcript-id-position
dict_train_bags, dict_train_labels = {}, {}

for index, row in tqdm(dat_train.iterrows(), total = len(dat_train)):
  if row['transcript_id_position'] not in dict_train_bags:
    dict_train_bags[row['transcript_id_position']] = []
  dict_train_bags[row['transcript_id_position']].append(index)

  if row['transcript_id_position'] not in dict_train_labels:
    dict_train_labels[row['transcript_id_position']] = row['label']

train_labels = np.array([v for v in dict_train_labels.values()])
train_bags = np.array([v for v in dict_train_bags.values()])
train_bags_number = np.array([k for k in dict_train_labels.keys()])

### try to reduce RAM here
del dict_train_bags
del dict_train_labels

## get testing bag instances, bag labels
## and bags corresponding to transcript-id-position
dict_test_bags, dict_test_labels = {}, {}

for index, row in tqdm(dat_test.iterrows(), total = len(dat_test)):
  if row['transcript_id_position'] not in dict_test_bags:
    dict_test_bags[row['transcript_id_position']] = []
  dict_test_bags[row['transcript_id_position']].append(index)

  if row['transcript_id_position'] not in dict_test_labels:
    dict_test_labels[row['transcript_id_position']] = row['label']

test_labels = np.array([v for v in dict_test_labels.values()])
test_bags = np.array([v for v in dict_test_bags.values()])
test_bags_number = np.array([k for k in dict_test_labels.keys()])

### try to reduce RAM here
del dict_test_bags
del dict_test_labels

'''Choose number of neighbours'''
n_neighbors_ref = 15
n_neighbors_cit = 5

'''Run models'''
# for each data set (CV), for each distance, perform 3 models
#for distance_mode, distance in zip(['max', 'avg', 'min'], [bag_distances_max, bag_distances_avg, bag_distances_min]):
for distance_mode, distance in zip(['max'], [bag_distances_max]):
  ### CHECK
  #print(f"fold is {i+1}, r is {r}, running distance is {distance_mode}")

  arr_temp = [i+1, dict_distance[distance_mode]]

  # kNN
  y_proba_test = get_knn_score(train_labels, test_labels, distance, n_neighbors_ref)
  y_pred_test = [int(proba > knn_threshold_value) for proba in y_proba_test]
  balanced_acc = balanced_accuracy_score(test_labels, y_pred_test)
  auc_roc = get_roc_auc(test_labels, y_proba_test)
  auc_pr = get_pr_auc(test_labels, y_proba_test)
  precision = precision_score(test_labels, y_pred_test, zero_division = 0.0)
  recall = recall_score(test_labels, y_pred_test, zero_division = 0.0)
  f1 = f1_score(test_labels, y_pred_test, zero_division = 0.0)
  tn, fp, fn, tp = confusion_matrix(test_labels, y_pred_test, labels = [0, 1]).ravel()
  res += [arr_temp + [dict_model['kNN'], balanced_acc, auc_roc, auc_pr, precision, recall, f1, tn, fp, fn, tp]]

  # citation kNN
  y_proba_test = get_citation_knn_score(train_labels, test_labels, distance, n_neighbors_ref)
  y_pred_test = [int(proba > cknn_threshold_value) for proba in y_proba_test]
  balanced_acc = balanced_accuracy_score(test_labels, y_pred_test)
  auc_roc = get_roc_auc(test_labels, y_proba_test)
  auc_pr = get_pr_auc(test_labels, y_proba_test)
  precision = precision_score(test_labels, y_pred_test, zero_division = 0.0)
  recall = recall_score(test_labels, y_pred_test, zero_division = 0.0)
  f1 = f1_score(test_labels, y_pred_test, zero_division = 0.0)
  tn, fp, fn, tp = confusion_matrix(test_labels, y_pred_test, labels = [0, 1]).ravel()
  res += [arr_temp + [dict_model['ckNN'], balanced_acc, auc_roc, auc_pr, precision, recall, f1, tn, fp, fn, tp]]

  # citation kNN with normalised
  y_proba_test = get_citation_knn_score_normalized(train_labels, test_labels, distance, n_neighbors_ref, n_neighbors_cit)
  y_pred_test = [int(proba > ncknn_threshold_value) for proba in y_proba_test]
  balanced_acc = balanced_accuracy_score(test_labels, y_pred_test)
  auc_roc = get_roc_auc(test_labels, y_proba_test)
  auc_pr = get_pr_auc(test_labels, y_proba_test)
  precision = precision_score(test_labels, y_pred_test, zero_division = 0.0)
  recall = recall_score(test_labels, y_pred_test, zero_division = 0.0)
  f1 = f1_score(test_labels, y_pred_test, zero_division = 0.0)
  tn, fp, fn, tp = confusion_matrix(test_labels, y_pred_test, labels = [0, 1]).ravel()
  res += [arr_temp + [dict_model['norm-ckNN'], balanced_acc, auc_roc, auc_pr, precision, recall, f1, tn, fp, fn, tp]]

'''Get results'''

df_res = pd.DataFrame(res, columns=['Validation on Fold','Distance Type', 'Model',
                                          'AUC ROC', 'AUC PRC', 'Precision', 'Recall', 'F1', 'Balanced Accuracy',
                                          'tn', 'fp', 'fn', 'tp'])

# Replace codes for distance metrics and model
df_res.loc[df_res['Distance Type'] == 0, 'Distance Type'] = 'max'
df_res.loc[df_res['Distance Type'] == 1, 'Distance Type'] = 'avg'
df_res.loc[df_res['Distance Type'] == 2, 'Distance Type'] = 'min'

df_res.loc[df_res['Model'] == 4, 'Model'] = 'kNN'
df_res.loc[df_res['Model'] == 5, 'Model'] = 'ckNN'
df_res.loc[df_res['Model'] == 6, 'Model'] = 'norm-ckNN'

df_res.sort_values(by=['Model', 'Distance Type', 'Validation on Fold']).to_csv(f"res_full_fold_{i+1}.csv")

df_res.groupby(['Model', 'Distance Type']).apply(lambda x: np.mean(x.iloc[:, 3:], axis = 0)).to_csv(f"res_averaged_fold_{i+1}.csv")

with open(f'train_fold_labels_fold_{i+1}.txt', 'w') as f:
  print(train_fold_labels, file=f)
