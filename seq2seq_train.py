import torch
import numpy as np
#np.set_printoptions(threshold=np.inf)
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
import seq2seq_Dataset
import seq2seq_model


#random.seed(312)
#torch.manual_seed(312)

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    #target = torch.LongTensor(target)
    return [data, target]

def random_sample(data, label,valid = False):
    # data shape : [*,25088]
    # label shape : [*]
    if valid == True:
        sample_num = data.shape[0]
        #print("sample_num = ",sample_num)
    else:
        sample_num = 350
    selected_idx = sorted(random.sample([i for i in range(0, data.size(0))], sample_num))
    sampled_data = [data[i] for i in selected_idx]
    sampled_label = [label[i] for i in selected_idx]
    sampled_data = torch.stack(sampled_data).unsqueeze(1)
    sampled_label = torch.stack(sampled_label).unsqueeze(1)
    length = [len(sampled_data)]

    length = torch.LongTensor(length) # torch.Size([6])
    #print(sampled_data.shape)
    #print(sampled_label.shape)
    #print(length.shape)

    return sampled_data, sampled_label, length

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def main():
    # parameters
    feature_size = 512 * 7 * 7
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 6

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/s2s"):
        os.makedirs("./logfile/s2s")

    # load my Dataset
    train_dataset = seq2seq_Dataset.seq2seq_Dataset(mode = "train")
    test_dataset = seq2seq_Dataset.seq2seq_Dataset(mode = "valid")

    print('the train_dataset has %d size.' % (len(train_dataset.data)))
    print('the valid_dataset has %d size.' % (len(test_dataset.data)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8,collate_fn=my_collate)

    print('the train_loader has %d size.' % (len(train_loader)))
    print('the test_loader has %d size.' % (len(test_loader)))

    model = seq2seq_model.seq2seq_model(feature_size)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        model = model.to(device)

    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas=(0.5, 0.999))
    load_checkpoint('./save/RNN-043.pth',model,optimizer)

    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    print("Starting training...")
    best_accuracy = -np.inf
    for epoch in range(num_epochs):
        model.train()
        print("Epoch:", epoch+1)
        epoch_train_loss = 0.0
        train_acc = 0.0

        if (epoch+1) == 50:
            optimizer.param_groups[0]['lr'] /= 2

        for i, (feature, label) in enumerate(train_loader):
            batch_size = len(train_loader)
            sampled_feature = torch.Tensor()
            sampled_label = torch.LongTensor()
            for i in range(batch_size):
                t_feature = feature[i]
                t_label = label[i]
                t_sampled_feature,t_sampled_label,lengths = random_sample(t_feature,t_label)
                sampled_feature = torch.cat((sampled_feature,t_sampled_feature),1)
                sampled_label = torch.cat((sampled_label,t_sampled_label),1)

            optimizer.zero_grad()
            sampled_feature = Variable(sampled_feature).to(device)
            sampled_label = Variable(sampled_label).to(device)
            lengths = Variable(lengths).to(device)

            # print("feature :",sampled_feature.shape)
            output = model(sampled_feature)
            # print("feature :",sampled_label.shape)

            # print("output :",output.shape)
            train_loss = 0
            for i in range(batch_size):
                t_output = output[:,i,:]
                t_sampled_label = sampled_label[:,i]
                loss = criterion(t_output, t_sampled_label)
                train_loss += loss
            train_loss /= batch_size
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

            # Accuracy
            acc = 0.
            for i in range(batch_size):
                t_output = output[:,i,:]
                t_sampled_label = sampled_label[:,i]
                t_output_label = torch.argmax(t_output,1).cpu()
                t_acc = np.mean((t_output_label == t_sampled_label.cpu()).numpy())
                acc += t_acc
            acc /= batch_size
            train_acc += acc
            print('Epoch [%d/%d], Iter [%d/%d] loss %.4f,Acc %.4f, LR = %.6f'
            %(epoch, num_epochs, i+1, len(train_loader), train_loss.item(),acc, optimizer.param_groups[0]['lr']))

        if (epoch) % 10 == 0:
            save_checkpoint('./save/s2s-%03i.pth' % (epoch) , model, optimizer)

        # testing
        with torch.no_grad():

            model.eval()
            epoch_test_loss = 0.0
            test_acc = 0.
            for i, (feature, label) in enumerate(test_loader):
                sampled_feature,sampled_label,lengths = random_sample(feature[0],label[0],valid = True)

                sampled_feature = Variable(sampled_feature).to(device)
                sampled_label = Variable(sampled_label).to(device)
                lengths = Variable(lengths).to(device)
                print("feature :",sampled_feature.shape)
                output = model(sampled_feature)
                sampled_label = sampled_label.view(-1)
                # print("output :",output.shape)
                # print("sampled_label :",sampled_label.shape)
                test_loss = criterion(output, sampled_label)
                epoch_test_loss += test_loss.item()
                # Accuracy
                output_label = torch.argmax(output,1).cpu()
                #if i==0:
                    #print(lengths)
                    #print(output_label)
                acc = np.mean((output_label == sampled_label.cpu()).numpy())
                print("Acc for %d : %.4f"%(lengths,acc))
                test_acc += acc
                """
                sampled_feature,sampled_label,lengths = random_sample(feature,label,valid = True)
                optimizer.zero_grad()
                sampled_feature = Variable(sampled_feature).to(device)
                sampled_label = Variable(sampled_label).to(device)
                lengths = Variable(lengths).to(device)
                # print("feature :",sampled_feature.shape)
                output = model(sampled_feature)
                test_loss = criterion(output, sampled_label)
                epoch_test_loss += test_loss.item()
                # Accuracy
                output_label = torch.argmax(output,1).cpu()
                acc = np.mean((output_label == sampled_label.cpu()).numpy())
                test_acc += acc
                """
        print('\n============\nEpoch [%d/%d] ,Train: Loss: %.4f | Acc: %.4f ,Validation: loss: %.4f | Acc: %.4f'
        %(epoch,num_epochs, epoch_train_loss/len(train_loader), train_acc/len(train_loader), epoch_test_loss/len(test_loader), test_acc/len(test_loader)))

        # save loss data
        train_loss_list.append(epoch_train_loss/len(train_loader))
        train_acc_list.append(train_acc/len(train_loader))
        test_loss_list.append(epoch_test_loss/len(test_loader))
        test_acc_list.append(test_acc/len(test_loader))

        if (test_acc/len(test_loader) > best_accuracy):
            best_accuracy = test_acc/len(test_loader)
            save_checkpoint('./save/s2s-RNN043-%03i-%.6f.pth' % (epoch,best_accuracy) , model, optimizer)
            print ('Save best model , test_acc = %.6f...' % (best_accuracy))

        print('-'*88)

    with open('./logfile/s2s/train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)
    with open('./logfile/s2s/train_acc.pkl', 'wb') as f:
        pickle.dump(train_acc_list, f)
    with open('./logfile/s2s/test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss_list, f)
    with open('./logfile/s2s/test_acc.pkl', 'wb') as f:
        pickle.dump(test_acc_list, f)


if __name__ == '__main__':
    main()
