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
from torch.utils import data
import skimage.io
import skimage
import os
import time
import pandas as pd
import random
import pickle
import CNN_Dataset
import CNN_model

random.seed(312)
torch.manual_seed(312)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    # parameters
    feature_size = 512 * 7 * 7
    learning_rate = 0.0001
    num_epochs = 51
    batch_size = 64

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/CNN"):
        os.makedirs("./logfile/CNN")

    # load my Dataset
    train_dataset = CNN_Dataset.CNN_Dataset(mode = "train")
    test_dataset = CNN_Dataset.CNN_Dataset(mode = "valid")

    print('the train_dataset has %d size.' % (len(train_dataset.data)))
    print('the valid_dataset has %d size.' % (len(test_dataset.data)))

    # Pre-train models
    CNN_pre_model = torchvision.models.vgg16(pretrained=True).features
    model = CNN_model.CNN_model(feature_size)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        CNN_pre_model = CNN_pre_model.to(device)
        model = model.to(device)

    ##################################################################
    #########        pre-train model vgg16
    ##################################################################
    # train_dataset = (num of data ,data/label ,frames ,3 ,224 ,224)

    CNN_pre_model.eval()
    train_features = []
    with torch.no_grad():
        for i in range(len(train_dataset.data)):
            print("\r%d/%d" %(i,len(train_dataset.data)),end="")
            input = train_dataset[i][0]
            input = input.to(device)
            tmp = CNN_pre_model(input).cpu().view(-1, feature_size)
            train_features.append(torch.mean(tmp,0).numpy())

        print(" Pre-train train_data finished!")

    test_features = []
    with torch.no_grad():
        for i in range(len(test_dataset.data)):
            print("\r%d/%d" %(i,len(test_dataset.data)),end="")
            input = test_dataset[i][0]
            input = input.to(device)
            tmp = CNN_pre_model(input).cpu().view(-1, feature_size)
            test_features.append(torch.mean(tmp,0).numpy())

        print(" Pre-train test_data finished!")

    # update dataset
    train_features = torch.Tensor(train_features)
    train_label = torch.LongTensor(train_dataset.label)
    train_features_dataset = data.TensorDataset(train_features, train_label)
    test_features = torch.Tensor(test_features)
    test_label = torch.LongTensor(test_dataset.label)
    test_features_dataset = data.TensorDataset(test_features, test_label)

    train_loader = DataLoader(train_features_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_features_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    print("Starting training...")
    best_accuracy = -np.inf
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        print("Epoch:", epoch+1)
        epoch_train_loss = 0.0
        train_acc = 0.0

        if (epoch+1) == 11:
            optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 20:
            optimizer.param_groups[0]['lr'] /= 2

        for i, (feature, label) in enumerate(train_loader):
            feature = Variable(feature).to(device)
            label = Variable(label).to(device)
            optimizer.zero_grad()
            output = model(feature)
            train_loss = criterion(output, label)
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

            # Accuracy
            output_label = torch.argmax(output,1).cpu()

            acc = np.mean((output_label == label.cpu()).numpy())
            train_acc += acc
            print('Epoch [%d/%d], Iter [%d/%d] loss %.4f,Acc %.4f, LR = %.6f'
            %(epoch, num_epochs, i+1, len(train_loader), train_loss.item(),acc, optimizer.param_groups[0]['lr']))

        if (epoch) % 10 == 0:
            save_checkpoint('./save/CNN-%03i.pth' % (epoch) , model, optimizer)

        # testing
        with torch.no_grad():

            model.eval()
            epoch_test_loss = 0.0
            test_acc = 0.
            for i, (feature, label) in enumerate(test_loader):
                feature = Variable(feature).to(device)
                label = Variable(label).to(device)
                output = model(feature)
                test_loss = criterion(output, label)
                predict = torch.argmax(output,1).cpu()
                acc = np.mean((predict == label.cpu()).numpy())
                epoch_test_loss += test_loss.item()
                test_acc += acc

        print('\n============\nEpoch [%d/%d] ,Train: Loss: %.4f | Acc: %.4f ,Validation: loss: %.4f | Acc: %.4f'
        %(epoch,num_epochs, epoch_train_loss/len(train_loader), train_acc/len(train_loader), epoch_test_loss/len(test_loader), test_acc/len(test_loader)))

        # save loss data
        train_loss_list.append(epoch_train_loss/len(train_loader))
        train_acc_list.append(train_acc/len(train_loader))
        test_loss_list.append(epoch_test_loss/len(test_loader))
        test_acc_list.append(test_acc/len(test_loader))

        if (test_acc/len(test_loader) > best_accuracy):
            best_accuracy = test_acc/len(test_loader)
            save_checkpoint('./save/CNN-%03i.pth' % (epoch) , model, optimizer)
            print ('Save best model , test_acc = %.6f)...' % (best_accuracy))

        print('-'*88)

    with open('./logfile/CNN/train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)
    with open('./logfile/CNN/train_acc.pkl', 'wb') as f:
        pickle.dump(train_acc_list, f)
    with open('./logfile/CNN/test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss_list, f)
    with open('./logfile/CNN/test_acc.pkl', 'wb') as f:
        pickle.dump(test_acc_list, f)


if __name__ == '__main__':
    main()
