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
from reader import readFullVideo
feature_size = 512 * 7 * 7

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
def test():
# test Reader
# input = video_path, video_category, video_name, downsample_factor=12, rescale_factor=1
    frames = readShortVideo("hw4_data/TrimmedVideos/video/train/",
                            "OP01-R01-PastaSalad", "OP01-R01-PastaSalad-66680-68130-F001597-F001639.mp4",
                            downsample_factor=12, rescale_factor=1)

    cc = frames[0]
    cc = cc.transpose(1,2,0)
    print(cc.shape)
    print(cc)
#plt.imshow(cc)
#plt.show()
############################################################
######      load training data & labels
############################################################
# -> data loading0
def load_train(video_path,model,device):
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

    model.eval()
    train_features = []
    with torch.no_grad():
        for i in range(len(train_data)):
            print("\r%d/%d" %(i,len(train_data)),end="")
            input = train_data[i]
            input = input.to(device)
            tmp = model(input).cpu().view(-1, feature_size)
            train_features.append(tmp)


        print(" Pre-train train_data finished!")


    with open("./hw4_data/RNN_Trimmed_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("./hw4_data/RNN_Trimmed_train_feature.pkl", "wb") as f:
        pickle.dump(train_features, f)
    with open("./hw4_data/RNN_Trimmed_train_label.pkl", "wb") as f:
        pickle.dump(train_label, f)
    return 0

############################################################
######      load validation data & labels
############################################################
# -> data loading
def load_test(video_path,model,device):
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

    model.eval()
    test_features = []
    with torch.no_grad():
        for i in range(len(test_data)):
            print("\r%d/%d" %(i,len(test_data)),end="")
            input = test_data[i]
            input = input.to(device)
            tmp = model(input).cpu().view(-1, feature_size)
            test_features.append(tmp)

        print(" Pre-train train_data finished!")


    with open("./hw4_data/RNN_Trimmed_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    with open("./hw4_data/RNN_Trimmed_test_feature.pkl", "wb") as f:
        pickle.dump(test_features, f)
    with open("./hw4_data/RNN_Trimmed_test_label.pkl", "wb") as f:
        pickle.dump(test_label, f)
    return 0

def main():
    # setting pretrain model
    CNN_pre_model = torchvision.models.vgg16(pretrained=True).features
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        CNN_pre_model = CNN_pre_model.to(device)

    train_path = "hw4_data/TrimmedVideos/video/train"
    valid_path = "hw4_data/TrimmedVideos/video/valid"
    print("Loading training data and pre-train model")
    load_train(train_path,CNN_pre_model,device)
    print("Loading testing data and pre-train model")
    load_test(valid_path,CNN_pre_model,device)

if __name__ == "__main__":
    main()
