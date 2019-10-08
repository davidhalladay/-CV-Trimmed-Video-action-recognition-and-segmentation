########################################################
## Author : Wan-Cyuan Fan
########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable


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
        return output

if __name__ == '__main__':
    model = RNN_model()
    print(model)
