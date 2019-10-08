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
import CNN_model

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def load_test_pred(video_path,gt_path,model_path):
    feature_size = 512 * 7 * 7

    CNN_pre_model = torchvision.models.vgg16(pretrained=True).features
    model = CNN_model.CNN_model(feature_size)
    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        CNN_pre_model = CNN_pre_model.to(device)
        model = model.to(device)
    load_checkpoint(model_path,model)
    CNN_pre_model.eval()

    # -> label loading
    test_label = pd.read_csv(gt_path)["Action_labels"]

    test_features = []
    category_path = sorted(os.listdir(video_path))
    with torch.no_grad():
        for category in category_path:
            mask = pd.read_csv(gt_path)["Video_category"] == category
            test_name = pd.read_csv(gt_path)[mask]["Video_name"]
            for i,video_name in enumerate(test_name):
                print("\r%d/%d" %(i,len(test_name)),end="")
                frames = readShortVideo(video_path,category, video_name,downsample_factor=12, rescale_factor=1)
                frames = Variable(torch.from_numpy(frames)).to(device)
                tmp = CNN_pre_model(frames).cpu().view(-1, feature_size)
                test_features.append(torch.mean(tmp,0).numpy())
            print("")
            print("Processing [%s] finished!"%(category))
            print("Pre-train finished!")

    test_features = torch.Tensor(test_features)

    model.eval()
    feature = Variable(test_features).to(device)
    output = model(feature)
    pred = torch.argmax(output,1).cpu()
    print(pred.shape)
    return test_features,pred ,test_label

def output_file(pred,gt_label,pred_folder):
    filename = os.path.join(pred_folder,"p1_valid.txt")
    file = open(filename,"w")
    # Accuracy
    count = 0.
    for i in range(len(pred)):
        if pred[i] == gt_label[i]:
            count += 1.
    print("Accuracy = ",(count)/len(pred))

    # write file
    for i in range(len(pred)):
        text = pred[i]
        file.write("%d\n"%(text))

    return True

def tSNE(test_features,pred):

    tSNE = manifold.TSNE(n_components = 2, init = 'pca', random_state = 312)
    feature_2D = tSNE.fit_transform(test_features)
    print(feature_2D.shape)
    plt.figure(figsize=(16, 16))
    plt.title('CNN features 2D')
    plt.scatter(feature_2D[:,0], feature_2D[:,1], c = pred, cmap = plt.cm.get_cmap("jet", 11))
    plt.colorbar(ticks=range(11))
    plt.clim(-0.5, 10.5)
    plt.savefig("./predict/CNN_2D.png")

    print("Plot CNN tSNE Done!")

def main():
    model_path = sys.argv[1]
    video_folder = sys.argv[2]
    ground_truth = sys.argv[3]
    pred_folder = sys.argv[4]

    test_features, pred,label = load_test_pred(video_folder,ground_truth,model_path)
    output_file(pred,label,pred_folder)
    # tSNE(test_features,pred)

if __name__ == '__main__':
    main()
