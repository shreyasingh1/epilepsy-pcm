import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 14 
hidden_size1 = 10
hidden_size2 = 10
num_classes = 1
num_epochs = 118
batch_size = 20
learning_rate = 0.001

## Make the Dataset
class PCMDataset(torch.utils.data.Dataset):

  def __init__(self, src_file):
    all_data = pd.read_csv(src_file, skiprows=0).to_numpy()  # strip IDs off

    self.x_data = torch.tensor(all_data[:,1:],dtype=torch.float32).to(device)
    self.y_data = torch.tensor(all_data[:,0],dtype=torch.float32).to(device)
    self.y_data = self.y_data.reshape(-1,1)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x_data[idx,:]  # idx rows, all 4 cols
    lbl = self.y_data[idx,:]    # idx rows, the 1 col
    sample = { 'predictors' : preds, 'target' : lbl }
    return sample

## Load the data
src1 = '/Users/david/Desktop/PCM_train.csv'
src2 = '/Users/david/Desktop/PCM_test.csv'

# get the training and testing dataset
train_dataset = PCMDataset(src1)
test_dataset = PCMDataset(src2)

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

# Fully connected neural network with two hidden layer
class Net(torch.nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2,num_classes):
    super(Net, self).__init__()
    self.hid1 = torch.nn.Linear(input_size, hidden_size1)  # 4-(8-8)-1
    self.hid2 = torch.nn.Linear(hidden_size1, hidden_size2)
    self.oupt = torch.nn.Linear(hidden_size2, num_classes)

    torch.nn.init.xavier_uniform_(self.hid1.weight)
    torch.nn.init.zeros_(self.hid1.bias)
    torch.nn.init.xavier_uniform_(self.hid2.weight)
    torch.nn.init.zeros_(self.hid2.bias)
    torch.nn.init.xavier_uniform_(self.oupt.weight)
    torch.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = torch.tanh(self.hid1(x)) 
    z = torch.tanh(self.hid2(z))
    z = torch.sigmoid(self.oupt(z))
    return z

model = Net(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss = 0.0  # sum of batch losses
    for (batch_idx, batch) in enumerate(train_loader):
        X = batch['predictors']  # [20,14]  inputs
        Y = batch['target']      # [20,1]  targets

        # Forward pass
        oupt = model(X)            # [20,1]  computed
        loss_val = criterion(oupt, Y)   # a tensor
        epoch_loss += loss_val.sum().item()  # accumulate

        # Backward and optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        if True:  
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

print("Done ")

# train on the test set and return accuracy
def accuracy(model, ds):
  # ds is a PyTorch Dataset
  # assumes model = model.eval()
  n_correct = 0; n_wrong = 0

  for i in range(len(ds)):
    inpts = ds[i]['predictors'] 
    target = ds[i]['target']
    with torch.no_grad():
      oupt = model(inpts)

    print("----------")
    print("input:    " + str(inpts))
    print("target:   " + str(target))
    print("computed: " + str(oupt))

    # avoid 'target == 1.0'
    if target < 0.5 and oupt < 0.5:
      n_correct += 1
      print("correct")
    elif target >= 0.5 and oupt >= 0.5:
      n_correct += 1
      print("correct")
    else:
      n_wrong += 1
      print("wrong")

    print("----------")
  return (n_correct * 1.0) / (n_correct + n_wrong)

print("\nBegin accuracy() test ")

model = model.eval()
acc = accuracy(model, test_dataset)
print("\nAccuracy = %0.4f" % acc)

print("\nEnd test ")