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
import RNN_Dataset
import RNN_model

random.seed(312)
torch.manual_seed(312)

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def single_pad_sequence(train_data, train_label):
    length = [len(x) for x in train_data]
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
    if not os.path.exists("./logfile/RNN"):
        os.makedirs("./logfile/RNN")

    # load my Dataset
    train_dataset = RNN_Dataset.RNN_Dataset(mode = "train")
    test_dataset = RNN_Dataset.RNN_Dataset(mode = "valid")

    print('the train_dataset has %d size.' % (len(train_dataset.data)))
    print('the valid_dataset has %d size.' % (len(test_dataset.data)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8,collate_fn=my_collate)

    print('the train_loader has %d size.' % (len(train_loader)))
    print('the test_loader has %d size.' % (len(test_loader)))

    model = RNN_model.RNN_model(feature_size)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        model = model.to(device)

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

        if (epoch+1) == 25:
            optimizer.param_groups[0]['lr'] /= 2

        for i, (feature, label) in enumerate(train_loader):
            padded_feature,padded_label,lengths = single_pad_sequence(feature,label)
            optimizer.zero_grad()
            padded_feature = Variable(padded_feature).to(device)
            padded_label = Variable(padded_label).to(device)
            lengths = Variable(lengths).to(device)
            #print(padded_feature.shape) # torch.Size([66, 64, 25088])
            #print(padded_label.shape) # torch.Size([64])
            #print(lengths.shape) # torch.Size([64])
            output = model(padded_feature,lengths)
            train_loss = criterion(output, padded_label)
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

            # Accuracy
            output_label = torch.argmax(output,1).cpu()

            acc = np.mean((output_label == padded_label.cpu()).numpy())
            train_acc += acc
            print('Epoch [%d/%d], Iter [%d/%d] loss %.4f,Acc %.4f, LR = %.6f'
            %(epoch, num_epochs, i+1, len(train_loader), train_loss.item(),acc, optimizer.param_groups[0]['lr']))

        if (epoch) % 10 == 0:
            save_checkpoint('./save/RNN-%03i.pth' % (epoch) , model, optimizer)

        # testing
        with torch.no_grad():

            model.eval()
            epoch_test_loss = 0.0
            test_acc = 0.
            for i, (feature, label) in enumerate(test_loader):
                padded_feature,padded_label,lengths = single_pad_sequence(feature,label)
                padded_feature = Variable(padded_feature).to(device)
                padded_label = Variable(padded_label).to(device)
                lengths = Variable(lengths).to(device)
                output = model(padded_feature,lengths)
                test_loss = criterion(output, padded_label)
                predict = torch.argmax(output,1).cpu()
                acc = np.mean((predict == padded_label.cpu()).numpy())
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
            save_checkpoint('./save/RNN-%03i.pth' % (epoch) , model, optimizer)
            print ('Save best model , test_acc = %.6f)...' % (best_accuracy))

        print('-'*88)
    """
    with open('./logfile/RNN/train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)
    with open('./logfile/RNN/train_acc.pkl', 'wb') as f:
        pickle.dump(train_acc_list, f)
    with open('./logfile/RNN/test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss_list, f)
    with open('./logfile/RNN/test_acc.pkl', 'wb') as f:
        pickle.dump(test_acc_list, f)
    """

if __name__ == '__main__':
    main()
