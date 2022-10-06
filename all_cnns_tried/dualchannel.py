import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms.functional as TorchFunctional

change = {
    'curr': [32, 'M', 64, 'M', 128, 128, 'M', 256, 'M', 512, 'M'],
}


class make_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(make_cnn, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(change['curr'])
        
        self.fcs = nn.Sequential(
            nn.LazyLinear(4096),
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
custommean = 0.7749, 0.6126, 0.6939
customstd = 0.2206, 0.2156, 0.2119
numepoch = 10
data_direction = os.getcwd()
transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
trainingdata = os.path.join(data_direction, r"splitdataset\train")

train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)
loader = DataLoader(train_dataset, batch_size = 100, shuffle=True)

model = make_cnn(in_channels=3, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.12]))
least = 999


def train():
    for epoch in range(numepoch):
        print("EPOCH DONE")
        for images, labels in loader:
            images = images.to(device)
            images = TorchFunctional.adjust_hue(images, -0.1)
            
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            global least

            if (loss < least):
                save()
                least = loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())

def save():
    torch.save(model.state_dict(), "bestdualnonorm.pth")

train()
