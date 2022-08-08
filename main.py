import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

custommean = 0.8132, 0.6343, 0.7334
customstd = 0.0807, 0.1310, 0.0968

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(custommean, customstd)])

data_direction = os.getcwd()
trainingdata = os.path.join(data_direction, "training")

train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)


def show_images(train_dataset):
    loader = torch.utils.data.DataLoader(train_dataset, batch_size = 50, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    print('labels: ', labels)


show_images(train_dataset)
