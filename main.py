import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

def fit(mse, num_epochs, optimizer):
    for i in range (num_epochs):
        for xb, yb in DataLoadTrainer:
            preds = model(xb)
            loss = mse(preds, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss)



model = nn.Linear(3, 2)

mse = F.mse_loss
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

dataset = TensorDataset(inputs, targets)
DataLoadTrainer = DataLoader(dataset, batch_size = 5, shuffle = True)

fit(mse, 500, optimizer)
