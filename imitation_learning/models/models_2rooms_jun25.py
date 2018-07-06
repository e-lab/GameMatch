from torchvision.models import alexnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# features + BatchNorm + LSTM + FC
class ALSTM2rooms(nn.Module):
    def __init__(self):
        super(ALSTM2rooms, self).__init__()

        # self.bn = nn.BatchNorm1d(9216)
        self.bn = nn.BatchNorm1d(2048)

        self.features = alexnet(pretrained=True).features

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, ceil_mode=False)

        # self.lstm = nn.LSTM(input_size=9216, hidden_size=512, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=1024, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=512, batch_first=True)

        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=128, bias=True), \
                nn.Linear(in_features=128, out_features=32, bias=True), nn.Softmax())


    def forward(self, x, states):
        # x = self.batchnorm(x)
        x = self.features(x)

        # Now x is 256 x 6 x 6
        x = self.conv6(x)
        # Now x is 512 x 4 x 4
        F.relu(x, inplace=True)
        x = self.maxpool(x)
        # now x is 512 x 2 x 2
        x = x.view(x.size(0), 512 * 4)
        # x = x.view(x.size(0), 256 * 6 * 6)

        x = self.bn(x)

        x = x.unsqueeze(0)
        # now the output should be of dimensions 1 x batch_size x 2048

        new_states = [None, None]
        x, new_states[0] = self.lstm1(x, states[0])
        x, new_states[1] = self.lstm2(x, states[1])

        # now the output should be of dimensions 1 x batch_size x 32
        x = x.squeeze()

        return self.classifier(x), new_states

