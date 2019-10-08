########################################################
## Author : Wan-Cyuan Fan
## Reference : https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, figsize = 64):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize = True , _stride = 2 , _padding = 1):
            layers = [nn.ConvTranspose2d(in_feat, out_feat,kernel_size=4, stride = _stride, padding = _padding, bias=False)]
            if normalize :
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(inplace = True))
            return layers

        self.model = nn.Sequential(
            # state : (1,1,100)
            *block(100, figsize * 8, normalize = False , _stride = 1 , _padding = 0),
            # state: (4,4,figsize * 8)
            *block(figsize * 8, figsize * 4),
            # state: (8,8,figsize * 4)
            *block(figsize * 4, figsize * 2),
            # state: (16,16,figsize * 2)
            *block(figsize * 2, figsize),
            # state: (32,32,figsize)
            nn.ConvTranspose2d(figsize , 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        output = img/2.0+0.5
        return output



class Discriminator(nn.Module):
    def __init__(self, figsize = 64):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, normalize = True , _stride = 2 , _padding = 1):
            layers = [nn.Conv2d(in_feat, out_feat,kernel_size=4, stride = _stride, padding = _padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            # state: (64,64,3)
            *block(3, figsize),
            # state: (32,32,figsize)
            *block(figsize, figsize * 2),
            # state: (16,16,figsize*2)
            *block(figsize * 2, figsize * 4),
            # state: (8,8,figsize*4)
            *block(figsize * 4, figsize * 8),
            # state: (4,4,figsize*2)
            nn.Conv2d(figsize * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        img = self.model(x)
        output = img.view(-1, 1).squeeze(1)
        return output
