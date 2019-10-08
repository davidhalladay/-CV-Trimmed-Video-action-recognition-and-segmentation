
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


def train_plot():
    file_path = "./logfile/s2s/test_loss.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)

        print(len(data))
        plt.figure(figsize=(8, 6))
        plt.title('Testing loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(data, label='Testing loss')
        plt.savefig("./logfile/s2s_test_loss.png")

if __name__ == "__main__":
    train_plot()
