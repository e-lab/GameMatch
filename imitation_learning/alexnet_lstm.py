from torchvision.models import alexnet

import torch
import torch.nn as nn
from torch.autograd import Variable


class ALSTM(nn.Module):
    def __init__(self):
        super(ALSTM, self).__init__()
        self.features = alexnet().features

        self.lstm = nn.LSTM(input_size=4096, hidden_size=1000, batch_first=True)

        self.classifier = alexnet().classifier[:-1]
        
        self.end = nn.Sequential(nn.Linear(in_features=1000, out_features=8, bias=True) \
            ,nn.Softmax())


    def forward(self, x, states):
        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.classifier(x)

        x = x.unsqueeze(0)

        # h0 = Variable(torch.zeros(1, 1, 1000))
        # c0 = Variable(torch.zeros(1, 1, 1000))

        # print(x.size())

        x, states = self.lstm(x, states)

        # now the output should be of dimensions 1 x batch_size x 8
        x = x.squeeze()

        return self.end(x), states
