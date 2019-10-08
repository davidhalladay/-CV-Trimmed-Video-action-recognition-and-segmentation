import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import sys
import string
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class GAN_Dataset(Dataset):
    def __init__(self, filepath):
        self.figsize = 64
        self.images = []
        self.file_list = os.listdir(filepath)
        self.file_list.sort()

        print("Load file from :" ,filepath)
        for i, file in enumerate(self.file_list):
            print("\r%d/%d" %(i,len(self.file_list)),end = "")
            img = cv2.imread(os.path.join(filepath, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.
            self.images.append(img)

        self.images = np.array(self.images)
        self.images = self.images.transpose(0, 3, 1, 2)
        self.images = torch.FloatTensor(self.images)
        print("")
        print("Loading file completed.")

        self.num_samples = len(self.images)

    def __getitem__(self, index):
        data = self.images[index]
        return data, data

    def __len__(self):
        return self.num_samples

def test():
    file_root = './hw3_data/face/train'
    train_dataset = GAN_Dataset(filepath = file_root )
    train_loader = DataLoader(train_dataset,batch_size=100,shuffle=True,num_workers=1)
    train_iter = iter(train_loader)
    print(len(train_loader.dataset))
    print(len(train_loader))
    for epoch in range(1):
        img,target = next(train_iter)

        #im = plt.imshow(img[0])
        plt.show()
        print(img.shape)
        print(target[0,1,1,:])
# main()

if __name__ == '__main__':
    test()
