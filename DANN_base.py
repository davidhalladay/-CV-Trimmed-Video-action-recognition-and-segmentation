import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import skimage.io
import skimage
import os
import time
import pandas as pd
import random
import pickle
import DANN_model
import DANN_Dataset

random.seed(312)
torch.manual_seed(312)

def save_checkpoint(checkpoint_path, model):#, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    # parameters
    s_train_file_root = "./hw3_data/digits/svhn/train"
    s_train_csv_root = "./hw3_data/digits/svhn/train.csv"
    s_test_file_root = "./hw3_data/digits/svhn/test"
    s_test_csv_root = "./hw3_data/digits/svhn/test.csv"

    learning_rate = 0.01
    num_epochs = 101
    batch_size = 500

    # load my Dataset
    s_train_dataset = DANN_Dataset.DANN_Dataset(filepath = s_train_file_root ,csvpath = s_train_csv_root)
    s_train_loader = DataLoader(s_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    s_test_dataset = DANN_Dataset.DANN_Dataset(filepath = s_test_file_root ,csvpath = s_test_csv_root)
    s_test_loader = DataLoader(s_test_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    print('the source dataset has %d size.' % (len(s_train_dataset)))
    print('the batch_size is %d' % (batch_size))

    # models setting
    feature_extractor = DANN_model.Feature_Extractor()
    label_predictor = DANN_model.Label_Predictor()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.to(device)
        label_predictor = label_predictor.to(device)

    # setup optimizer
    optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                            {'params': label_predictor.parameters()}], lr= learning_rate)

    #Lossfunction
    L_criterion = nn.NLLLoss()

    print("Starting training...")

    for epoch in range(num_epochs):
        feature_extractor.train()
        label_predictor.train()

        print("Epoch:", epoch+1)
        len_dataloader = len(s_train_loader)

        epoch_L_loss = 0.0

        if (epoch+1) == 11:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        #    F_optimizer.param_groups[0]['lr'] /= 2
        #    L_optimizer.param_groups[0]['lr'] /= 2
        #    D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 20:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        #    F_optimizer.param_groups[0]['lr'] /= 2
        #    L_optimizer.param_groups[0]['lr'] /= 2
        #    D_optimizer.param_groups[0]['lr'] /= 2

        for i, source_data in enumerate(s_train_loader):
            source_img, source_label = source_data
            source_img = Variable(source_img).to(device)
            source_label = Variable(source_label).to(device)

            # train the feature_extractor
            optimizer.zero_grad()

            source_feature = feature_extractor(source_img)

            # Label_Predictor network
            src_label_output = label_predictor(source_feature)
            _, src_pred_arg = torch.max(src_label_output,1)
            src_acc = np.mean(np.array(src_pred_arg.cpu()) == np.array(source_label.cpu()))
            loss = L_criterion(src_label_output, source_label)

            epoch_L_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print('Epoch [%d/%d], Iter [%d/%d] loss %.4f , LR = %.6f'
                %(epoch, num_epochs, i+1, len_dataloader, loss.item(), optimizer.param_groups[0]['lr']))

        # epoch done
        print('-'*88)

    save_checkpoint('./save/DANN_base_F_svhn.pth' , feature_extractor)
    save_checkpoint('./save/DANN_base_L_svhn.pth' , label_predictor)

    # shuffle
if __name__ == '__main__':
    main()
