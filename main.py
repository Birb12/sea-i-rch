import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets

def savemodel():
    path = "./savedmodel.pth"
    torch.save(model.state_dict(), path) # state dict is current weights and biases

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, 50) 
        self.layer2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
num_output = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 1

dataset = torchvision.datasets.MNIST(root='dataset/', train=True, transform=torchvision.transforms.ToTensor()) # placeholder
dataload = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size, num_classes=num_output).to(device) 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data, targets in dataload:
        data = data.to(device=device)
        targets = targets.to(device=device)
        data = data.reshape(data.shape[0], -1) 
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)


