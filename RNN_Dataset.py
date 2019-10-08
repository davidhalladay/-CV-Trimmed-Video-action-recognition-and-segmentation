######################################
# Author : Wan-Cyuan Fan
######################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import os.path
import sys
import string
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class RNN_Dataset(Dataset):

    def __init__(self,mode):
        print('Initialize the Dataset...')
        self.mode = mode
        if self.mode == "train":
            with open("./hw4_data/RNN_Trimmed_train_feature.pkl", "rb") as f:
                self.data = pickle.load(f)
            with open("./hw4_data/RNN_Trimmed_train_label.pkl", "rb") as f:
                self.label = pickle.load(f)

        if self.mode == "valid":
            with open("./hw4_data/RNN_Trimmed_test_feature.pkl", "rb") as f:
                self.data = pickle.load(f)
            with open("./hw4_data/RNN_Trimmed_test_label.pkl", "rb") as f:
                self.label = pickle.load(f)
        print('Ending Initialization...')
        self.num_samples = len(self.data)

    def __getitem__(self,idx):
        img = self.data[idx]
        target = self.label[idx]

        return img,target

    def __len__(self):
        return self.num_samples

def test():
    train_dataset = RNN_Dataset(mode = "valid")
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    print(len(train_loader.dataset))
    print(len(train_loader))
    for epoch in range(1):
        img,target = next(train_iter)
        print("img shape : ",img.shape)
        print("target shape : ",target.shape)
        # img shape :  torch.Size([1, 7, 224, 224, 3])
        # target shape :  torch.Size([1])
        print("target is : ",target[0])
        print(img)

if __name__ == "__main__":
    test()
