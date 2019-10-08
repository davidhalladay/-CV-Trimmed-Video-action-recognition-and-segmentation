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
import DSN_model
from DSN_lossfunction import mse,simse ,DiffLoss
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
    t_train_file_root = "./hw3_data/digits/usps/train"
    t_train_csv_root = "./hw3_data/digits/usps/train.csv"
    t_test_file_root = "./hw3_data/digits/usps/test"
    t_test_csv_root = "./hw3_data/digits/usps/test.csv"

    learning_rate = 0.01
    num_epochs = 100
    batch_size = 32

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/DSN"):
        os.makedirs("./logfile/DSN")
    if not os.path.exists("./save_imgs"):
        os.makedirs("./save_imgs")
    if not os.path.exists("./save_imgs/DSN"):
        os.makedirs("./save_imgs/DSN")

    # load my Dataset
    s_train_dataset = DANN_Dataset.DANN_Dataset(filepath = s_train_file_root ,csvpath = s_train_csv_root)
    s_train_loader = DataLoader(s_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    s_test_dataset = DANN_Dataset.DANN_Dataset(filepath = s_test_file_root ,csvpath = s_test_csv_root)
    s_test_loader = DataLoader(s_test_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    t_train_dataset = DANN_Dataset.DANN_Dataset(filepath = t_train_file_root ,csvpath = t_train_csv_root)
    t_train_loader = DataLoader(t_train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    t_test_dataset = DANN_Dataset.DANN_Dataset(filepath = t_test_file_root ,csvpath = t_test_csv_root)
    t_test_loader = DataLoader(t_test_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    print('the source dataset has %d size.' % (len(s_train_dataset)))
    print('the target dataset has %d size.' % (len(t_train_dataset)))
    print('the batch_size is %d' % (batch_size))

    # models setting
    net = DSN_model.DSN()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    # setup optimizer
    optimizer = optim.SGD(net.parameters(), lr= learning_rate,momentum = 0.9, weight_decay = 1e-6)

    #Lossfunction
    class_criterion = nn.CrossEntropyLoss()
    recon_criterion_1 = mse()
    recon_criterion_2 = simse()
    diff_criterion = DiffLoss()
    similarity_criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    my_step = 0
    active_step = 10000
    for epoch in range(num_epochs):
        net.train()

        print("Epoch:", epoch+1)
        len_dataloader = min(len(s_train_loader), len(t_train_loader))
        total_step = len_dataloader * num_epochs

        if (epoch+1) == 11:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        if (epoch+1) == 20:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        if (epoch+1) == 60:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2

        for i, (source_data, target_data) in enumerate(zip(s_train_loader,t_train_loader)):
            source_img, source_label = source_data
            target_img, target_label = target_data
            batch_size = len(source_img)
            from_source_labels = torch.zeros(batch_size).type(torch.LongTensor)
            batch_size = len(target_img)
            from_target_labels = torch.ones(batch_size).type(torch.LongTensor)

            source_img = Variable(source_img).to(device)
            source_label = Variable(source_label).to(device)
            target_img = Variable(target_img).to(device)
            target_label = Variable(target_label).to(device)

            from_source_labels = Variable(from_source_labels).to(device)
            from_target_labels = Variable(from_target_labels).to(device)

            # colculate the lambda_
            p = (i + len_dataloader * epoch)/(len_dataloader * num_epochs)
            lambda_ = 2.0 / (1. + np.exp(-10 * p)) - 1.0

            # train the target_img
            net.zero_grad()
            loss = 0.0

            # active domain loss
            if my_step > active_step:
                result = net(target_img,'target','all',lambda_)
                target_privte_feature, target_share_feature, target_domain_label, target_recon_img = result
                target_label_loss = 0.3 * similarity_criterion(target_domain_label,from_target_labels)
                loss += target_label_loss

            else:
                target_label_loss = Variable(torch.zeros(1).float()).cuda()
                result = net(target_img,'target','all')
                target_privte_feature, target_share_feature, _, target_recon_img = result

            target_diff = 0.08 * diff_criterion(target_privte_feature, target_share_feature)
            loss += target_diff
            target_recon_1 = 0.01 * recon_criterion_1(target_recon_img, target_img)
            loss += target_recon_1
            target_recon_2 = 0.01 * recon_criterion_2(target_recon_img, target_img)
            loss += target_recon_2

            loss.backward()
            optimizer.step()

            # train the source_img
            net.zero_grad()
            loss = 0.0

            # active domain loss
            if my_step > active_step:
                result = net(source_img,'source','all',lambda_)
                source_privte_feature, source_share_feature, source_domain_label, source_class_label, source_recon_feature = result
                source_label_loss = 0.3 * similarity_criterion(source_domain_label,from_source_labels)
                loss += source_label_loss
            else:
                source_label_loss = Variable(torch.zeros(1).float()).cuda()
                result = net(source_img,'source','all')
                source_privte_feature, source_share_feature, _, source_class_label, source_recon_feature = result

            source_class_loss = class_criterion(source_class_label, source_label)
            loss += source_class_loss
            source_diff = 0.08 * diff_criterion(source_privte_feature, source_share_feature)
            loss += source_diff
            source_recon_1 = 0.01 * recon_criterion_1(source_recon_feature, source_img)
            loss += source_recon_1
            source_recon_2 = 0.01 * recon_criterion_2(source_recon_feature, source_img)
            loss += source_recon_2

            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print('Epoch [%d/%d], Iter [%d/%d] target_label_loss %f, source_label_loss %f , LR = %.4f'
                %(epoch, num_epochs, i+1, len_dataloader, target_label_loss, source_label_loss, optimizer.param_groups[0]['lr']))
                print("target_diff : %f , target_recon : %f , source_class_loss : %f, source_diff : %f , source_recon_1 : %f"
                %(target_diff,target_recon_1,source_class_loss,source_diff,source_recon_1))

        if (epoch) % 50 == 0:
            save_checkpoint('./save/DSN-%03i.pth' % (epoch) , net)
        my_step += 1
        # epoch done
        print('-'*88)

    # shuffle
if __name__ == '__main__':
    main()
