from torchvision.models import alexnet

import torch
import torch.nn as nn
from torch.autograd import Variable

# add fully-connected layer w/ BatchNorm
class AFC(nn.Module):
    def __init__(self):
        super(AFC, self).__init__()

        self.features = alexnet().features

        self.classifier = alexnet().classifier
        self.bn = nn.BatchNorm1d(1000)
        
        self.end = nn.Sequential(nn.Linear(in_features=1000, out_features=32, bias=True) \
            ,nn.Softmax())


    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.classifier(x)

        x = self.bn(x)

        return self.end(x)


# add BatchNorm + LSTM + FC
class ALSTM(nn.Module):
    def __init__(self):
        super(ALSTM, self).__init__()

        # self.batchnorm = nn.BatchNorm2d(3)
        self.bn = nn.BatchNorm1d(4096)

        self.features = alexnet().features

        self.lstm = nn.LSTM(input_size=4096, hidden_size=1000, batch_first=True)

        self.classifier = alexnet().classifier[:-1]
        
        self.end = nn.Sequential(nn.Linear(in_features=1000, out_features=32, bias=True) \
            ,nn.Softmax())


    def forward(self, x, states):
        # x = self.batchnorm(x)
        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.classifier(x)

        x = self.bn(x)

        x = x.unsqueeze(0)

        # h0 = Variable(torch.zeros(1, 1, 1000))
        # c0 = Variable(torch.zeros(1, 1, 1000))

        # print(x.size())

        x, states = self.lstm(x, states)

        # now the output should be of dimensions 1 x batch_size x 32
        x = x.squeeze()

        return self.end(x), states


# add BatchNorm + LSTM
class ALSTM1(nn.Module):
    def __init__(self):
        super(ALSTM1, self).__init__()

        # self.batchnorm = nn.BatchNorm2d(3)

        self.features = alexnet().features

        self.lstm = nn.LSTM(input_size=1000, hidden_size=32, batch_first=True)

        self.classifier = alexnet().classifier

        self.bn = nn.BatchNorm1d(1000)
        
        self.sm = nn.Softmax()


    def forward(self, x, states):
        # x = self.batchnorm(x)
        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.classifier(x)

        x = self.bn(x)

        x = x.unsqueeze(0)

        x, states = self.lstm(x, states)

        # now the output should be of dimensions 1 x batch_size x 32
        x = x.squeeze()

        return self.sm(x), states
