import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# on a dataset with 250k images, this took my machine (Ryzen 5) 45 minutes to compute at 99% CPU; be warned

path = os.path.join(os.getcwd(), "training")
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(path, transform)
loader = DataLoader(dataset, batch_size = 32, shuffle = True)

def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_images = 0

    for images, _ in loader:
        image_count_in_batch = images.size(0)
        images = images.view(image_count_in_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += image_count_in_batch

    mean /= total_images 
    std /= total_images
    print(mean)
    print(std)

    return mean, std

get_mean_and_std(loader)
