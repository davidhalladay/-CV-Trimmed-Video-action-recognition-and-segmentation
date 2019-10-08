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

class RNN_model(nn.Module):
    def __init__(self, input_size = 512*7*7, hidden_size=512):
        super(RNN_model, self).__init__()
        self.hidden_size =  hidden_size
        self.LSTM = nn.LSTM(input_size, self.hidden_size, num_layers=2,dropout=(0.5))
        self.model = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, int((self.hidden_size/2.))),
            nn.BatchNorm1d(int((self.hidden_size/2.))),
            nn.Linear(int(self.hidden_size/2.), 11),
            nn.Softmax(1)
        )

    def forward(self, X, lengths):
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths)
        tmp, (h_n,c_n) = self.LSTM(packed_X, None) # output: (seq_len, batch, hidden size)
        #print(len(tmp[1]))
        #print(len(h_n)) # (2,64,512)

        h_output = h_n[-1]
        #print(len(h_output))
        #print(h_output.shape) # (64,512)
        output = self.model(h_output)
        #print(output.shape)
        return output,h_output

def single_pad_sequence(train_data, train_label):
    length = [len(train_data)]
    length_index = np.argsort(length)[::-1] # longest to shortest

    # sorting
    train_data_sorted = []
    for i in length_index:
        train_data_sorted.append(train_data[i])
    lengths = [len(x) for x in train_data_sorted]
    padded_sequence = nn.utils.rnn.pad_sequence(train_data_sorted)
    # padded_sequence = padded_sequence.transpose(0,1) #(batch_size , longest length ,feature_size)
    # print("padded_sequence size :",padded_sequence.shape)
    label = torch.LongTensor(np.array(train_label)[length_index])
    lengths = torch.LongTensor(np.array(lengths))
    return padded_sequence, label, lengths

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def load_test_pred(video_path,gt_path,model_path):
    feature_size = 512 * 7 * 7

    CNN_pre_model = torchvision.models.vgg16(pretrained=True).features
    model = RNN_model(feature_size)
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
    # test_label = pd.read_csv(gt_path)["Action_labels"]

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
                test_features.append(tmp)
            print("")
            print("Processing [%s] finished!"%(category))
            print("Pre-train finished!")


    with torch.no_grad():
        RNN_feature = []
        preds = []
        model.eval()
        for i in range(0, len(test_features)):
            padded_feature,lengths = test_features[i], [test_features[i].shape[0]] # padded_label, test_label[i]
            padded_feature = Variable(padded_feature).to(device).unsqueeze(1)
            lengths = torch.LongTensor(lengths)
            #print(padded_feature.shape)
            #print(padded_label)
            lengths = Variable(lengths).to(device)
            output ,hidden= model(padded_feature,lengths)
            pred = torch.argmax(output,1).cpu()
            preds.append(pred)
            RNN_feature.append(hidden.cpu().data.numpy().reshape(-1))

    RNN_feature = np.array(RNN_feature)
    preds = np.array(preds)
    print(pred.shape)
    return RNN_feature,preds # ,test_label

def output_file(pred,pred_folder): # gt_label
    filename = os.path.join(pred_folder,"p2_result.txt")
    file = open(filename,"w")
    # Accuracy
    #count = 0.
    #for i in range(len(pred)):
    #    if pred[i] == gt_label[i]:
    #        count += 1.
    #print("Accuracy = ",(count)/len(pred))

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
    plt.title('RNN features 2D')
    plt.scatter(feature_2D[:,0], feature_2D[:,1], c = pred, cmap = plt.cm.get_cmap("jet", 11))
    plt.colorbar(ticks=range(11))
    plt.clim(-0.5, 10.5)
    plt.savefig("./predict/RNN_2D.png")

    print("Plot RNN tSNE Done!")


def main():
    model_path = sys.argv[1]
    video_folder = sys.argv[2]
    ground_truth = sys.argv[3]
    pred_folder = sys.argv[4]

    test_features, pred = load_test_pred(video_folder,ground_truth,model_path) #label
    output_file(pred,pred_folder) # label
    #tSNE(test_features,pred)

if __name__ == '__main__':
    main()
