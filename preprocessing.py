from os import listdir
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import os.path
import sys
import string
import matplotlib
import matplotlib.pyplot as plt
# self-defined module
from reader import readShortVideo
from reader import getVideoList

#video_path = "hw4_data/TrimmedVideos/video/valid"
#category_path = sorted(os.listdir(video_path))
#for category in category_path:
#    video_name = sorted(os.listdir(os.path.join(video_path,category)))
#    for i,video in enumerate(video_name):
#        print(video)
#train_label = pd.read_csv("hw4_data/TrimmedVideos/label/gt_valid.csv")
#train_label = train_label.sort_values(["Video_name"])["Action_labels"]
#train_label = torch.LongTensor(train_label)
#train_l = train_label["Action_labels"]
#print(train_label)
#train_filename_sorted = []
#test_filename_sorted = []

# test Reader
# input = video_path, video_category, video_name, downsample_factor=12, rescale_factor=1
#frames = readShortVideo("hw4_data/TrimmedVideos/video/train/",
#                        "OP01-R01-PastaSalad", "OP01-R01-PastaSalad-66680-68130-F001597-F001639.mp4",
#                        downsample_factor=12, rescale_factor=1)
#
#cc = frames[0]
#cc = cc.transpose(1,2,0)
#print(cc.shape)
#plt.imshow(cc)
#plt.show()
############################################################
######      load training data & labels
############################################################
# -> data loading
def load_train(video_path):
    all_video_frames = []
    category_path = sorted(os.listdir(video_path))
    for category in category_path:
        video_name = sorted(os.listdir(os.path.join(video_path,category)))
        for i,video in enumerate(video_name):
            print("\r%d/%d" %(i,len(video_name)),end="")
            frames = readShortVideo(video_path,category, video,downsample_factor=12, rescale_factor=1)
            all_video_frames.append(torch.from_numpy(frames))

        print("")
        print("Processing [%s] finished!"%(category))
    # -> label loading

    train_data = all_video_frames
    train_label = pd.read_csv("hw4_data/TrimmedVideos/label/gt_train.csv")
    train_label = train_label.sort_values(["Video_name"])["Action_labels"]
    train_label = torch.LongTensor(train_label)

    with open("./hw4_data/Trimmed_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("./hw4_data/Trimmed_train_label.pkl", "wb") as f:
        pickle.dump(train_label, f)
    return 0

############################################################
######      load validation data & labels
############################################################
# -> data loading
def load_test(video_path):
    all_video_frames = []
    category_path = sorted(os.listdir(video_path))
    for category in category_path:
        video_name = sorted(os.listdir(os.path.join(video_path,category)))
        for i,video in enumerate(video_name):
            print("\r%d/%d" %(i,len(video_name)),end="")
            frames = readShortVideo(video_path,category, video,downsample_factor=12, rescale_factor=1)
            all_video_frames.append(torch.from_numpy(frames))

        print("")
        print("Processing [%s] finished!"%(category))
    # -> label loading
    test_data = all_video_frames
    test_label = pd.read_csv("hw4_data/TrimmedVideos/label/gt_valid.csv")
    test_label = test_label.sort_values(["Video_name"])["Action_labels"]
    test_label = torch.LongTensor(test_label)

    with open("./hw4_data/Trimmed_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    with open("./hw4_data/Trimmed_test_label.pkl", "wb") as f:
        pickle.dump(test_label, f)
    return 0

def main():
    train_path = "hw4_data/TrimmedVideos/video/train"
    valid_path = "hw4_data/TrimmedVideos/video/valid"
    load_train(train_path)
    load_test(valid_path)

if __name__ == "__main__":
    main()
