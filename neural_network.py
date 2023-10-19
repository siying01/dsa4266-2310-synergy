# import dependencies
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, precision_recall_curve, auc
from scipy.stats import mode

df = pd.read_csv("training_data.csv")


# first fold
i = 1

train_data = df.loc[df['folds']!=i].reset_index()
test_data = df.loc[df['folds']==i].reset_index()

X_train = np.array(train_data.loc[:, 'time_1':'mean_current_3'])
X_test = np.array(test_data.loc[:, 'time_1':'mean_current_3'])

train_grouped = train_data.groupby(['transcript_id', 'transcript_position']).groups
train_indices = [np.array(indices) for indices in train_grouped.values()]
train_bags = np.array(train_indices, dtype=object)
train_labels = df.loc[[bag[0] for bag in train_bags], 'label'].values

test_grouped = test_data.groupby(['transcript_id', 'transcript_position']).groups
test_indices = [np.array(indices) for indices in test_grouped.values()]
test_bags = np.array(test_indices, dtype=object)
test_labels = df.loc[[bag[0] for bag in test_bags], 'label'].values


torch.random.manual_seed(0)

class nnMI(nn.Module):
  
  def __init__(self, encoder, decoder, mode='max'):
    super(nnMI, self).__init__()
    if mode not in ('max', 'mean'):
      raise ValueError("Invalid mode {}, must be one of max or mean".format(mode))
    self.mode = mode
    self.encoder = encoder
    self.decoder = decoder 
    
  def forward(self, x, indices):
    x = self.encoder(x)
    if self.mode == 'max':
      x = torch.stack([torch.max(x[idx], axis=0).values for idx in indices])
    else:
      x = torch.stack([torch.mean(x[idx], axis=0).values for idx in indices])
    x = self.decoder(x)
    return x


# neural network architecture
device = 'cpu'

encoder = nn.Sequential(*[nn.Linear(9, 32), nn.ReLU(),
                          nn.Linear(32, 9), nn.ReLU()])

decoder = nn.Sequential(*[nn.Linear(9, 32), nn.ReLU(),
                          nn.Linear(32, 9), nn.ReLU(),
                          nn.Linear(9, 1)])

nn_mi = nnMI(encoder, decoder, mode='max').to(device)


# calculate weights for the minority class
classes, counts = np.unique(train_labels, return_counts=True)
class_weights = 1./torch.tensor(counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()

weights = torch.zeros(len(train_labels))
weights[np.argwhere(train_labels == 0)] = class_weights[0]
weights[np.argwhere(train_labels == 1)] = class_weights[1]


# define loss function
criterion = nn.BCEWithLogitsLoss(weight=weights)
optimizer = Adam(nn_mi.parameters(), lr=0.0001)    #0.001

# torch training loop
n_epochs = 100
for i in range(n_epochs):
  nn_mi.train()
  optimizer.zero_grad()
  output = nn_mi(torch.Tensor(X_train), train_bags)
  loss = criterion(output.flatten(), torch.Tensor(train_labels))
  loss.backward()
  optimizer.step()
  print(i)

nn_mi.eval()


# functions to calculate metrics
def get_roc_auc(y_true, y_pred):
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc

def get_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

with torch.no_grad():
  y_pred_nn_test = torch.sigmoid(nn_mi(torch.Tensor(X_test).to(device), test_bags)).detach().cpu().numpy().flatten()
  print("ROC AUC: {}".format(get_roc_auc(test_labels, y_pred_nn_test)))
  print("PR AUC: {}".format(get_pr_auc(test_labels, y_pred_nn_test)))
  print("Accuracy Score: {}".format(accuracy_score(test_labels, y_pred_nn_test >= 0.5)))
  print("Balanced Accuracy Score: {}".format(balanced_accuracy_score(test_labels, y_pred_nn_test >= 0.5)))
  
  
with open('result.txt', 'w') as f:
    f.write("ROC AUC: {}".format(get_roc_auc(test_labels, y_pred_nn_test)))
    f.write('\n')
    f.write("PR AUC: {}".format(get_pr_auc(test_labels, y_pred_nn_test)))
    f.write('\n')
    f.write("Accuracy Score: {}".format(accuracy_score(test_labels, y_pred_nn_test >= 0.5)))
    f.write('\n')
    f.write("Balanced Accuracy Score: {}".format(balanced_accuracy_score(test_labels, y_pred_nn_test >= 0.5)))
    