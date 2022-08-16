import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

VGG = {
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG['VGG19'])
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
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


model = VGG_net(in_channels=3,num_classes=2)

        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

custommean = 0.7455, 0.5832, 0.6751
customstd = 0.2143, 0.2036, 0.2033

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.RandomRotation(degrees=(-20, +20)), transforms.Normalize(custommean, customstd)])
outputs = ("1", "0")


numepoch = 8
data_direction = os.getcwd()
trainingdata = os.path.join(data_direction, "test-train")
least_loss = 1

train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)
loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)

def train():
    for epoch in range(numepoch):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            global least_loss

            outputs = model(images)
            loss = criterion(outputs, labels)

            if (loss < least_loss):
                least_loss = loss
                save()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())
    


def show_images(train_dataset):
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    print('labels: ', labels)

def save():

    torch.save(model.state_dict(), 'current-best.pth')
    print("model saved")


train()
