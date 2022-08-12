import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ConvNetwork(nn.Module):
    def __init__(self):

        super(ConvNetwork, self).__init__()

        self.layer1 = nn.Conv2d(3, 6, 5)
        self.poollayer = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1296, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):

        x = self.poollayer(F.relu(self.layer1(x)))
        x = self.poollayer(F.relu(self.layer2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

custommean = 0.7455, 0.5832, 0.6751
customstd = 0.2143, 0.2036, 0.2033

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((50, 50)), transforms.Normalize(custommean, customstd), transforms.RandomRotation(degrees=(-20, +20))])
model = ConvNetwork().to(device)
classes = ("0", "1")
model.load_state_dict(torch.load('current-best.pth'))
data_direction = os.getcwd()
trainingdata = os.path.join(data_direction, "test-train")
train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)
loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)

def classify(model, imagetransforms, imagepath, classes):
    model = model.eval()
    image = Image.open(imagepath)
    image = imagetransforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print(predicted.item())

classify(model, transform, imagepath=r'C:\Users\mined\Desktop\projects\torch\test-train\1\internet3.png', classes=classes)
