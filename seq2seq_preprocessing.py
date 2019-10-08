from os import listdir
import pandas as pd
import numpy as np
#np.set_printoptions(threshold=np.inf)
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
    model = torchvision.models.vgg16(pretrained=True).features
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        model = model.to(device)
# test Reader
# input = video_path, video_category, video_name, downsample_factor=12, rescale_factor=1
    frames = readFullVideo("hw4_data/FullLengthVideos/videos/train/","OP01-R01-PastaSalad")
    print(len(frames))
    train_feature = []
    with torch.no_grad():
        for i in range(len(frames)):
            train_input = frames[i]
            train_input = train_input.unsqueeze(0).cuda()
            features = model(train_input).detach().cpu().numpy().reshape(-1,feature_size)
            train_feature.append(features)

    print(train_feature[0].shape)
    print(train_feature[0])
    #img = plt.imshow(frames[0])
    #plt.show()
    #print(frames[0])
    # -> label loading
    train_labels = []
    train_label_path = "./hw4_data/FullLengthVideos/labels/train"
    labels = sorted(os.listdir(train_label_path))
    for i,label in enumerate(labels):
        file_path = os.path.join(train_label_path,label)
        with open(file_path, 'r') as f:
            lines = [int(line.strip()) for line in f.readlines()]
            train_labels.append(lines)
    train_labels = np.array(train_labels)
    return 0
#cc = frames[0]
#cc = cc.transpose(1,2,0)
#print(cc.shape)
#plt.imshow(cc)
#plt.show()
############################################################
######      load training data & labels
############################################################

# -> data loading
def load_train(video_path,model,device):
    all_video_feature = []
    category_path = sorted(os.listdir(video_path))
    model.eval()
    for i,category in enumerate(category_path):
        frames = readFullVideo(video_path, category)
        print("")
        train_feature = []
        with torch.no_grad():
            for i in range(len(frames)):
                train_input = frames[i]
                train_input = torch.Tensor(train_input).unsqueeze(0).to(device)
                features = model(train_input).detach().cpu().numpy().reshape(-1,feature_size)
                train_feature.append(features)

        all_video_feature.append(torch.from_numpy(np.vstack(train_feature)))

    print("Feature shape : ",all_video_feature[0].shape)
    print("Processing finished!")

    train_features = all_video_feature

    with open("./hw4_data/Seq_Full_train_feature.pkl", "wb") as f:
        pickle.dump(train_features, f)

    print("Save Seq_Full_train_feature.pkl")
    # -> label loading
    train_label = []
    train_label_path = "./hw4_data/FullLengthVideos/labels/train"
    labels = sorted(os.listdir(train_label_path))
    for i,label in enumerate(labels):
        file_path = os.path.join(train_label_path,label)
        with open(file_path, 'r') as f:
            lines = [int(line.strip()) for line in f.readlines()]
            train_label.append(torch.LongTensor(lines))

    with open("./hw4_data/Seq_Full_train_label.pkl", "wb") as f:
        pickle.dump(train_label, f)

    print("Save Seq_Full_train_label.pkl")
    return 0

############################################################
######      load validation data & labels
############################################################
# -> data loading
def load_test(video_path,model,device):
    all_video_feature = []
    category_path = sorted(os.listdir(video_path))
    model.eval()
    for i,category in enumerate(category_path):
        frames = readFullVideo(video_path, category)
        print("")
        test_feature = []
        with torch.no_grad():
            for i in range(len(frames)):
                test_input = frames[i]
                test_input = torch.Tensor(test_input).unsqueeze(0).to(device)
                features = model(test_input).detach().cpu().numpy().reshape(-1,feature_size)
                test_feature.append(features)

        all_video_feature.append(torch.from_numpy(np.vstack(test_feature)))

    print("Feature shape : ",all_video_feature[-1].shape)
    print("Processing finished!")

    test_features = all_video_feature

    with open("./hw4_data/Seq_Full_test_feature.pkl", "wb") as f:
        pickle.dump(test_features, f)

    print("Save Seq_Full_test_feature.pkl")

    # -> label loading
    test_label = []
    test_label_path = "./hw4_data/FullLengthVideos/labels/valid"
    labels = sorted(os.listdir(test_label_path))
    for i,label in enumerate(labels):
        file_path = os.path.join(test_label_path,label)
        with open(file_path, 'r') as f:
            lines = [int(line.strip()) for line in f.readlines()]
            test_label.append(torch.LongTensor(lines))

    with open("./hw4_data/Seq_Full_test_label.pkl", "wb") as f:
        pickle.dump(test_label, f)

    print("Save Seq_Full_test_label.pkl")
    return 0

def main():
    # setting pretrain model
    pre_model = torchvision.models.vgg16(pretrained=True).features
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        pre_model = pre_model.to(device)

    train_path = "hw4_data/FullLengthVideos/videos/train"
    valid_path = "hw4_data/FullLengthVideos/videos/valid"
    print("Loading training data and pre-train model")
    load_train(train_path,pre_model,device)
    print("Loading testing data and pre-train model")
    load_test(valid_path,pre_model,device)

if __name__ == "__main__":
    main()
