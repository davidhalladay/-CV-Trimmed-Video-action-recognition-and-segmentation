from os import listdir
from sklearn import manifold, datasets
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import os.path
import sys
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# self-defined module
from reader import readShortVideo
from reader import getVideoList
from reader import readFullVideo

class seq2seq_model(nn.Module):
    def __init__(self, feature_size= 512*7*7 , hidden_size=512):
        super(seq2seq_model, self).__init__()
        self.hidden_size =  hidden_size
        self.LSTM = nn.LSTM(feature_size, self.hidden_size, num_layers=2,dropout=(0.5))
        self.model = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, int((self.hidden_size/2.))),
            nn.BatchNorm1d(int((self.hidden_size/2.))),
            nn.Linear(int(self.hidden_size/2.), 11),
            nn.Softmax(1)
        )

    def forward(self, X):
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths)
        output_seq = []
        tmp, (h_n,c_n) = self.LSTM(X, None) # output: (seq_len, batch, hidden size)
        # print(tmp.shape) # torch.Size([500, 6, 512])
        # print(h_n[1].shape) # torch.Size([6, 512])
        for i in range(tmp.shape[0]):
            #print(tmp[i].shape)
            #tmp_dd = tmp[i].unsqueeze(0)
            category = self.model(tmp[i])
            output_seq.append(category)

        output = torch.stack(output_seq) # tensor([[2,2,2,2,21,2]])
        output = output.squeeze(1)
        return output


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def load_test_pred(video_path,model_path):

    feature_size = 512 * 7 * 7
    CNN_pre_model = torchvision.models.vgg16(pretrained=True).features
    model = seq2seq_model(feature_size)
    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        CNN_pre_model = CNN_pre_model.to(device)
        model = model.to(device)
    load_checkpoint(model_path,model)
    CNN_pre_model.eval()


    all_video_feature = []
    category_path = sorted(os.listdir(video_path))
    CNN_pre_model.eval()
    for i,category in enumerate(category_path):
        frames = readFullVideo(video_path, category)
        print("")
        train_feature = []
        with torch.no_grad():
            for i in range(len(frames)):
                train_input = frames[i]
                train_input = torch.Tensor(train_input).unsqueeze(0).to(device)
                features = CNN_pre_model(train_input).detach().cpu().numpy().reshape(-1,feature_size)
                train_feature.append(features)
        all_video_feature.append(torch.from_numpy(np.vstack(train_feature)))

    # eval model
    all_pred =[]
    model.eval()
    with torch.no_grad():
        for i ,feature in enumerate(all_video_feature):
            feature = feature.unsqueeze(1)
            #print(feature.shape)
            feature = Variable(feature).to(device)
            output = model(feature)
            output_label = torch.argmax(output,1).cpu()
            all_pred.append(output_label)
    #print(len(all_pred))
    return all_pred,category_path

def Accuracy(all_pred,category_path):
    test_label = []
    test_label_path = "./hw4_data/FullLengthVideos/labels/valid"
    labels = sorted(os.listdir(test_label_path))
    for i,label in enumerate(labels):
        file_path = os.path.join(test_label_path,label)
        with open(file_path, 'r') as f:
            lines = [int(line.strip()) for line in f.readlines()]
            test_label.append(torch.LongTensor(lines))

    # Accuracy
    for idx in range(len(all_pred)):
        category = category_path[idx]
        pred = all_pred[idx]
        gt_label = test_label[idx]
        count = 0.
        for i in range(len(pred)):
            if pred[i] == gt_label[i]:
                count += 1.
        print(category+" accuracy = ",(count)/len(pred))
    return True

def output_file(all_pred,pred_folder,category_path):

    # write file
    for i in range(len(all_pred)):
        category = category_path[i]
        filename = os.path.join(pred_folder,"%s.txt"%(category))
        file = open(filename,"w")
        pred = all_pred[i]
        for j in range(len(pred)):
            text = pred[j]
            file.write("%d\n"%(text))
    print("Output file done!")
    return True

def plot_img(all_pred):

    test_label = []
    test_label_path = "./hw4_data/FullLengthVideos/labels/valid"
    labels = sorted(os.listdir(test_label_path))
    for i,label in enumerate(labels):
        file_path = os.path.join(test_label_path,label)
        with open(file_path, 'r') as f:
            lines = [int(line.strip()) for line in f.readlines()]
            test_label.append(torch.LongTensor(lines))

    # select OP01-R02-TurkeySandwich
    gt_label = test_label[2].numpy()
    pred = all_pred[2]

    plt.figure(figsize=(16, 16))
    plt.title('Prediction(upper) vs Ground truth(lower)')
    colors = plt.cm.get_cmap('Paired',11).colors
    cmap = matplotlib.colors.ListedColormap([colors[i] for i in pred])
    bounds = [i for i in range(len(pred))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = matplotlib.colorbar.ColorbarBase(plt.subplot(211), cmap=cmap,
                                               norm=norm,
                                               boundaries=bounds,
                                               spacing='proportional',
                                               orientation='horizontal')



    colors = plt.cm.get_cmap('Paired',11).colors
    cmap = matplotlib.colors.ListedColormap([colors[i] for i in gt_label])
    bounds = [i for i in range(len(gt_label))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = matplotlib.colorbar.ColorbarBase(plt.subplot(212), cmap=cmap,
                                               norm=norm,
                                               boundaries=bounds,
                                               spacing='proportional',
                                               orientation='horizontal')

    plt.savefig("./predict/groundtruth.png")


    print("Plot seq2seq Done!")


def main():
    model_path = sys.argv[1]
    video_folder = sys.argv[2]
    pred_folder = sys.argv[3]

    all_pred,category_path = load_test_pred(video_folder,model_path)
    output_file(all_pred,pred_folder,category_path)
    Accuracy(all_pred,category_path)
    #plot_img(all_pred)

if __name__ == '__main__':
    main()
