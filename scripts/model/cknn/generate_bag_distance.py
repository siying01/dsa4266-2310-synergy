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

'''Distance metric'''

def get_hausdorff_distance(X_train, X_test, train_bags, test_bags, mode='max'):
  all_bag_distances = []

  for test_bag in tqdm(test_bags, total=len(test_bags)):
    bag_distances = []

    for train_bag in train_bags:
      test_bag_instances = pairwise_distances(X_test[test_bag], X_train[train_bag]) # extract all the instances related to a particular test_bag, and get the distances from the test_bag instances to all train_bag instances

      if mode == 'max':
        test_train_distance = np.max(np.min(test_bag_instances, axis=1))
        train_test_distance = np.max(np.min(test_bag_instances.transpose(), axis=1))
        bag_distances.append(max(test_train_distance, train_test_distance))

      elif mode =='avg':
        test_train_distance = np.min(test_bag_instances, axis=1)
        train_test_distance = np.min(test_bag_instances.transpose(), axis=1)
        bag_distances.append(np.mean(np.concatenate([train_test_distance, test_train_distance])))

      elif mode == 'min':
        test_train_distance = np.min(np.min(test_bag_instances, axis=1))
        train_test_distance = np.min(np.min(test_bag_instances.transpose(), axis=1))
        bag_distances.append(max(test_train_distance, train_test_distance))

    all_bag_distances.append(np.array(bag_distances).reshape(1, -1))
  all_bag_distances = np.concatenate(all_bag_distances, axis=0)
  return all_bag_distances

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

'''Compute bag distances'''
bag_distances_max = get_hausdorff_distance(X_train, X_test, train_bags, test_bags, mode='max')
#bag_distances_avg = get_hausdorff_distance(X_train, X_test, train_bags, test_bags, mode='avg')
#bag_distances_min = get_hausdorff_distance(X_train, X_test, train_bags, test_bags, mode='min')

'''Save bag distances'''
np.save(f"bag_distances_max_validation_fold_{i+1}.npy", bag_distances_max)
