########################################################
## Author : Wan-Cyuan Fan
########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable

feature_size = 1024 * 8

class CNN_model(torch.nn.Module):
    def __init__(self, feature_size = 512 * 7 * 7):
        super(CNN_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 11),
            nn.Softmax(1)
        )

    def forward(self, x):
        output = self.model(x)
        return output

if __name__ == '__main__':
    model = CNN_model()
    print(model)
