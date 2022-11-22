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
from math import ceil


change = {
    'curr': [32, 'M', 64, 'M', 128, 128, 'M', 256, 'M', 512, 'M'], # to allow for easy changes to the architecture
}


class make_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(make_cnn, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(change['curr'])
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)



class dualchannel(nn.Module): 
    def __init__(self):
        super(dualchannel, self).__init__()
        self.ConvNetA = make_cnn()
        self.ConvNetB = make_cnn()
    def forward(self, x):
        x1 = self.ConvNetA.forward(x)
        x2 = self.ConvNetB.forward(x)

        x = torch.cat((x1, x2), dim=1) 
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
custommean = 0.8132, 0.6343, 0.7334 # calculated in normalization.py
customstd = 0.0807, 0.1310, 0.0968
numepoch = 8
data_direction = os.getcwd()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(custommean, customstd)])

trainingdata = os.path.join(data_direction, "test-train") # change to fit your directory
classes = ("0", "1")
 
train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)
loader = torch.utils.data.DataLoader(train_dataset, batch_size = 100, shuffle=True)

model = make_cnn(in_channels=3, num_classes=2).to(device)
model.load_state_dict(torch.load("bestdual.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

def classify(model, imagetransforms, imagepath, classes): # run one image through the model
    model = model.eval()
    image = Image.open(imagepath)
    image = imagetransforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

classify(model, transform, imagepath=r'C:\Users\mined\Desktop\projects\torch\test-train\1\8863_idx5_x1201_y901_class1.png', classes=classes)
