import pandas as pd
import numpy as np
import os
import splitfolders
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def split_path(path):
    if os.path.exists('Dataset/dataset_splits'):
        return
    else:
        splitfolders.ratio(path,output = 'Dataset/dataset_splits',seed = 1337,ratio = (0.8,0.1,0.1))

def dataset_loader(train_path,val_path,test_path):
    train_transform = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),#How to resize image
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485,0.456,0.406],
                                        std = [0.229,0.224,0.225])])

    val_transform = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485,0.456,0.406],
                                        std = [0.229,0.224,0.225] )])

    test_transform = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485,0.456,0.406],
                                        std = [0.229,0.224,0.225])])

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    print(f"Training data samples:{len(train_dataset)}\nValidation data samples:\
         {len(val_dataset)}\nTesting data samples: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    print('Data DUELY HANDLED')

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)
# train_loader, val_loader, _, train_size, val_size, _

split_path ("Dataset/rps-cv-images")
    