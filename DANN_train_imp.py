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
import DANN_model_imp
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
    s_train_file_root = "./hw3_data/digits/mnistm/train"
    s_train_csv_root = "./hw3_data/digits/mnistm/train.csv"
    s_test_file_root = "./hw3_data/digits/mnistm/test"
    s_test_csv_root = "./hw3_data/digits/mnistm/test.csv"
    t_train_file_root = "./hw3_data/digits/svhn/train"
    t_train_csv_root = "./hw3_data/digits/svhn/train.csv"
    t_test_file_root = "./hw3_data/digits/svhn/test"
    t_test_csv_root = "./hw3_data/digits/svhn/test.csv"

    learning_rate = 0.01
    num_epochs = 200
    batch_size = 500

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/DANN"):
        os.makedirs("./logfile/DANN")
    if not os.path.exists("./save_imgs"):
        os.makedirs("./save_imgs")
    if not os.path.exists("./save_imgs/DANN"):
        os.makedirs("./save_imgs/DANN")

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
    feature_extractor = DANN_model_imp.Feature_Extractor()
    label_predictor = DANN_model_imp.Label_Predictor()
    domain_classifier = DANN_model_imp.Domain_classifier()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.to(device)
        label_predictor = label_predictor.to(device)
        domain_classifier = domain_classifier.to(device)

    # setup optimizer
    optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                            {'params': label_predictor.parameters()},
                            {'params': domain_classifier.parameters()}], lr= learning_rate)

    #Lossfunction
    D_criterion = nn.BCELoss()
    L_criterion = nn.NLLLoss()

    D_loss_list = []
    L_loss_list = []
    sum_src_acc_list = []
    sum_trg_acc_list = []
    sum_label_acc_loist = []
    sum_test_acc_list = []
    print("Starting training...")

    for epoch in range(num_epochs):
        feature_extractor.train()
        label_predictor.train()
        domain_classifier.train()

        print("Epoch:", epoch+1)
        len_dataloader = min(len(s_train_loader), len(t_train_loader))

        epoch_D_loss = 0.0
        epoch_L_loss = 0.0
        sum_src_acc = 0.0
        sum_trg_acc = 0.0
        sum_label_acc = 0.0
        sum_test_acc = 0.0

        if (epoch+1) == 11:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        #    F_optimizer.param_groups[0]['lr'] /= 2
        #    L_optimizer.param_groups[0]['lr'] /= 2
        #    D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 50:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        if (epoch+1) == 80:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        if (epoch+1) == 150:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        if (epoch+1) == 250:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        if (epoch+1) == 400:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        if (epoch+1) == 500:
            for optimizer_t in optimizer.param_groups:
                optimizer_t['lr'] /= 2
        #    F_optimizer.param_groups[0]['lr'] /= 2
        #    L_optimizer.param_groups[0]['lr'] /= 2
        #    D_optimizer.param_groups[0]['lr'] /= 2

        for i, (source_data, target_data) in enumerate(zip(s_train_loader,t_train_loader)):
            source_img, source_label = source_data
            target_img, target_label = target_data
            batch_size = len(source_img)
            from_source_labels = torch.zeros(batch_size)
            batch_size = len(target_img)
            from_target_labels = torch.ones(batch_size)

            source_img = Variable(source_img).to(device)
            source_label = Variable(source_label).to(device)
            target_img = Variable(target_img).to(device)
            target_label = Variable(target_label).to(device)

            from_source_labels = Variable(from_source_labels).to(device)
            from_target_labels = Variable(from_target_labels).to(device)

            # colculate the lambda_
            p = (i + len_dataloader * epoch)/(len_dataloader * num_epochs)
            lambda_ = 2.0 / (1. + np.exp(-10 * p)) - 1.0

            # train the feature_extractor
            optimizer.zero_grad()

            source_feature = feature_extractor(source_img)
            target_feature = feature_extractor(target_img)

            # Label_Predictor network
            src_label_output = label_predictor(source_feature)
            _, src_pred_arg = torch.max(src_label_output,1)
            src_acc = np.mean(np.array(src_pred_arg.cpu()) == np.array(source_label.cpu()))
            L_loss = L_criterion(src_label_output, source_label)

            # Domain_classifier network with source domain
            src_domain_output = domain_classifier(source_feature,lambda_).view(-1)
            src_domain_acc = np.mean(np.array(src_domain_output.detach().cpu()) >= 0.5)
            D_loss_src = D_criterion(src_domain_output, from_source_labels)

            # Domain_classifier network with target domain
            trg_domain_output = domain_classifier(target_feature,lambda_).view(-1)
            # print("trg_domain_output : ",trg_domain_output)
            trg_domain_acc = np.mean(np.array(trg_domain_output.detach().cpu()) < 0.5)

            D_loss_trg = D_criterion(trg_domain_output, from_target_labels)
            # print("D_loss_src : ",D_loss_src)
            # print("D_loss_trg : ",D_loss_trg)
            D_loss = D_loss_src + D_loss_trg
            loss = L_loss + D_loss

            epoch_D_loss += D_loss.item()
            epoch_L_loss += L_loss.item()
            sum_src_acc += src_domain_acc
            sum_trg_acc += trg_domain_acc
            sum_label_acc += src_acc

            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print('Epoch [%d/%d], Iter [%d/%d] D_loss %.4f, L_loss %.4f , LR = %.6f'
                %(epoch, num_epochs, i+1, len_dataloader, D_loss.item(), L_loss.item(), optimizer.param_groups[0]['lr']))
                print("label_acc : %.4f , Domain src_acc : %.4f , Domain trg_acc : %.4f" %(src_acc,src_domain_acc,trg_domain_acc))

        if (epoch) % 200 == 0:
            save_checkpoint('./save/DANN-F-%03i.pth' % (epoch) , feature_extractor)
            save_checkpoint('./save/DANN-L-%03i.pth' % (epoch) , label_predictor)
            save_checkpoint('./save/DANN-D-%03i.pth' % (epoch) , domain_classifier)

        # testing
        """
        for i, (source_data, target_data) in enumerate(zip(s_train_loader,t_train_loader)):
            source_img, source_label = source_data
            target_img, target_label = target_data
            source_img = Variable(source_img).to(device)
            source_label = Variable(source_label).to(device)
            target_img = Variable(target_img).to(device)
            target_label = Variable(target_label).to(device)

            feature_extractor.eval()
            label_predictor.eval()
            domain_classifier.eval()

            target_feature = feature_extractor(target_img)
            trg_label_output = label_predictor(target_feature)
            _, trg_preds = torch.max(trg_label_output,1)
            trg_acc = np.mean(np.array(trg_preds.cpu()) == np.array(target_label.cpu()))

            sum_test_acc += trg_acc

        print("Test avg trg_acc : " %(sum_test_acc/len_dataloader))
        """
        # save loss data
        print("training D Loss:", epoch_D_loss/len_dataloader)
        print("training L Loss:", epoch_L_loss/len_dataloader)
        D_loss_list.append(epoch_D_loss/len_dataloader)
        L_loss_list.append(epoch_L_loss/len_dataloader)
        sum_src_acc_list.append(sum_src_acc/len_dataloader)
        sum_trg_acc_list.append(sum_trg_acc/len_dataloader)
        sum_label_acc_loist.append(sum_label_acc/len_dataloader)
        #sum_test_acc_list.append(sum_test_acc/len_dataloader)

        # epoch done
        print('-'*88)

    with open('./logfile/DANN/D_loss.pkl', 'wb') as f:
        pickle.dump(D_loss_list, f)
    with open('./logfile/DANN/L_loss.pkl', 'wb') as f:
        pickle.dump(L_loss_list, f)
    with open('./logfile/DANN/src_acc.pkl', 'wb') as f:
        pickle.dump(sum_src_acc_list, f)
    with open('./logfile/DANN/trg_acc.pkl', 'wb') as f:
        pickle.dump(sum_trg_acc_list, f)
    with open('./logfile/DANN/label_acc.pkl', 'wb') as f:
        pickle.dump(sum_label_acc_loist, f)
    #with open('./logfile/DANN/test_acc.pkl', 'wb') as f:
    #    pickle.dump(sum_test_acc_list, f)

    # shuffle
if __name__ == '__main__':
    main()
