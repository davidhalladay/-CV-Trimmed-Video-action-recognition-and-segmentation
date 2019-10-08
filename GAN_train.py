import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import skimage.io
import skimage
import os
import time
import pandas as pd
import random
import pickle
import GAN_model
from GAN_Dataset import GAN_Dataset

random.seed(312)
torch.manual_seed(312)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    # parameters
    file_root = "./hw3_data/face/train"
    learning_rate = 0.001
    num_epochs = 151
    batch_size = 100
    input_dim = 100

    # fixed input for model eval
    rand_inputs = Variable(torch.randn(32,input_dim, 1, 1), volatile=True)

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/GAN"):
        os.makedirs("./logfile/GAN")
    if not os.path.exists("./save_imgs"):
        os.makedirs("./save_imgs")
    if not os.path.exists("./save_imgs/GAN"):
        os.makedirs("./save_imgs/GAN")

    # load my Dataset
    train_dataset = GAN_Dataset(filepath = file_root )
    train_loader = DataLoader(train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    print('the dataset has %d size.' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))

    # models setting
    G_model = GAN_model.Generator()
    D_model = GAN_model.Discriminator()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        rand_inputs = rand_inputs.to(device)
        G_model ,D_model = G_model.to(device) ,D_model.to(device)

    # setup optimizer
    G_optimizer = optim.Adam(G_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    D_loss_list = []
    G_loss_list = []
    D_real_acc_list = []
    D_fake_acc_list = []

    print("Starting training...")

    for epoch in range(num_epochs):
        print("Epoch:", epoch+1)
        G_model.train()
        D_model.train()
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        D_real_total_acc = 0.0
        D_fake_total_acc = 0.0

        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 20:
            G_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        for i, (data, _) in enumerate(train_loader):
            batch_size = len(data)
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)
            data = Variable(data).to(device)
            real_labels = Variable(real_labels).to(device)
            fake_labels = Variable(fake_labels).to(device)

            # train the Generator
            G_model.zero_grad()
            z = torch.randn(batch_size, input_dim, 1, 1)
            z = Variable(z).to(device)
            fake_img = G_model(z)
            outputs = D_model(fake_img)
            G_loss = criterion(outputs, real_labels)
            epoch_G_loss += G_loss.item()
            G_loss.backward()
            G_optimizer.step()

            # train the Discriminator
            # BCE_Loss(x, y) = - y * log(D(x)) - (1-y) * log(1 - D(x))
            # real images , real_labels == 1

            D_model.zero_grad()
            outputs = D_model(data)
            D_real_loss = criterion(outputs, real_labels)
            D_real_acc = np.mean((outputs > 0.5).cpu().data.numpy())

            # fake images
            # First term of the loss is always zero since fake_labels == 0
            # we don't want to colculate the G gradient
            outputs = D_model(fake_img.detach())
            D_fake_loss = criterion(outputs, fake_labels)
            D_fake_acc = np.mean((outputs < 0.5).cpu().data.numpy())
            D_loss = (D_real_loss + D_fake_loss) / 2.

            D_loss.backward()
            D_optimizer.step()

            D_real_total_acc += D_real_acc
            D_fake_total_acc += D_fake_acc
            epoch_D_loss += D_loss.item()
            print('Epoch [%d/%d], Iter [%d/%d] G loss %.4f, D loss %.4f , LR = %.6f'
            %(epoch, num_epochs, i+1, len(train_loader), G_loss.item(), D_loss.item(), learning_rate))

        if (epoch) % 30 == 0:
            save_checkpoint('./save/GAN-G-%03i.pth' % (epoch) , G_model, G_optimizer)
            save_checkpoint('./save/GAN-D-%03i.pth' % (epoch) , D_model, D_optimizer)

        # save loss data
        D_loss_list.append(epoch_D_loss/len(train_loader.dataset))
        G_loss_list.append(epoch_G_loss/len(train_loader.dataset))
        D_real_acc_list.append(D_real_total_acc/len(train_loader))
        D_fake_acc_list.append(D_fake_total_acc/len(train_loader))

        # testing
        G_model.eval()
        test_output = G_model(rand_inputs)
        torchvision.utils.save_image(test_output.cpu().data,
                                './save_imgs/GAN/%03d_0.jpg' %(epoch+1), nrow=8)
        # epoch done
        print('-'*88)
    #
    with open('./logfile/GAN/D_loss.pkl', 'wb') as f:
        pickle.dump(D_loss_list, f)
    with open('./logfile/GAN/G_loss.pkl', 'wb') as f:
        pickle.dump(G_loss_list, f)
    with open('./logfile/GAN/D_real_acc.pkl', 'wb') as f:
        pickle.dump(D_real_acc_list, f)
    with open('./logfile/GAN/D_fake_acc.pkl', 'wb') as f:
        pickle.dump(D_fake_acc_list, f)

    # shuffle
if __name__ == '__main__':
    main()
