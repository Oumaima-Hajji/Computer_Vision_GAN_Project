from unittest import loader
from sklearn.utils import shuffle
from torchaudio import datasets
import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np 


train_dataset_path = 'train data path'
test_dataset_path = 'train data path'

os.listdir('path')
training_dataset_path = 'path'
training_transforms = transforms.Compose([transforms.Resize(250,250),
transforms.ToTensor()
])

train_dataset= torchvision.datasets.ImageFolder(root= training_dataset_path, transforms = training_transforms)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size = 32, shuffle = False)

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_image_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        print(images.shape)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        print(images.shape)

        mean += images.view(image_count_in_a_batch, images.size(1), -1) 
        std += images.std(2).sum(0)
        total_image_count += image_count_in_a_batch

    mean /= total_image_count
    std /= total_image_count

    return mean, std





def show_transformed_images(dataset):
    '''
    Show the Transformed images
    '''
    Loader = torch._utils.dataloader = torch.utils.DataLoader(dataset, batch_size=1, suffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow()


    
if __name__ == "__main__ " :
    get_mean_and_std(train_loader)
    