########################################################
## Author : Wan-Cyuan Fan
########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class seq2seq_model(nn.Module):
    def __init__(self, feature_size= 512*7*7 , hidden_size=512):
        super(seq2seq_model, self).__init__()
        self.hidden_size =  hidden_size
        self.LSTM = nn.LSTM(feature_size, self.hidden_size, num_layers=2,dropout=(0.5))
        self.model = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, int((self.hidden_size/2.))),
            nn.BatchNorm1d(int((self.hidden_size/2.))),
            nn.Linear(int(self.hidden_size/2.), 11),
            nn.Softmax(1)
        )

    def forward(self, X):
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths)
        output_seq = []
        tmp, (h_n,c_n) = self.LSTM(X, None) # output: (seq_len, batch, hidden size)
        # print(tmp.shape) # torch.Size([500, 6, 512])
        # print(h_n[1].shape) # torch.Size([6, 512])
        for i in range(tmp.shape[0]):
            #print(tmp[i].shape)
            #tmp_dd = tmp[i].unsqueeze(0)
            category = self.model(tmp[i])
            output_seq.append(category)

        output = torch.stack(output_seq) # tensor([[2,2,2,2,21,2]])
        output = output.squeeze(1)
        return output

if __name__ == '__main__':
    model = Seq2seq_model()
    print(model)
