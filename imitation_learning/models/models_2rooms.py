from torchvision.models import alexnet

import torch
import torch.nn as nn
from torch.autograd import Variable

# features + BatchNorm + LSTM + FC
class ALSTM2(nn.Module):
    def __init__(self):
        super(ALSTM2, self).__init__()

        # self.batchnorm = nn.BatchNorm2d(3)
        self.bn = nn.BatchNorm1d(9216)

        self.features = alexnet(pretrained=True).features

        self.lstm = nn.LSTM(input_size=9216, hidden_size=512, batch_first=True)

        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=128), \
                nn.Linear(in_features=128, out_features=32, bias=True), nn.Softmax())


    def forward(self, x, states):
        # x = self.batchnorm(x)
        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.bn(x)

        x = x.unsqueeze(0)

        x, states = self.lstm(x, states)

        # now the output should be of dimensions 1 x batch_size x 32
        x = x.squeeze()

        return self.classifier(x), states

