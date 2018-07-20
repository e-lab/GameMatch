# Basic networks used for 1 room scenarios
# Author: Ruihang Du
# email: du113@purdue.edu

from torchvision.models import alexnet

import torch
import torch.nn as nn
from torch.autograd import Variable

# use pretrained alexnet feature-extractor + batchnorm + custom fully-connected layers and softmax 
class AFC(nn.Module):
    def __init__(self):
        super(AFC, self).__init__()

        self.features = alexnet(pretrained=True).features

        # self.classifier = alexnet(pretrained=True).classifier[:-1]
        self.bn = nn.BatchNorm1d(9216)
        
        self.end = nn.Sequential(nn.Linear(in_features=9216, out_features=4096, bias=True), \
                nn.Linear(in_features=4096, out_features=2048, bias=True), \
                nn.Linear(in_features=2048, out_features=512, bias=True), \
                nn.Linear(in_features=512, out_features=32, bias=True), \
                nn.Softmax())


    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        # x = self.classifier(x)

        x = self.bn(x)

        return self.end(x)


# pretrained AlexNet feature-extractor + batchnorm + LSTM + FC
class ALSTM(nn.Module):
    def __init__(self):
        super(ALSTM, self).__init__()

        # self.batchnorm = nn.BatchNorm2d(3)
        self.bn = nn.BatchNorm1d(9216)

        self.features = alexnet(pretrained=True).features

        self.lstm = nn.LSTM(input_size=9216, hidden_size=4096, batch_first=True)

        # self.classifier = alexnet(pretrained=True).classifier[:-1]
        
        self.end = nn.Sequential(
                nn.Linear(in_features=4096, out_features=2048, bias=True), \
                nn.Linear(in_features=2048, out_features=512, bias=True), \
                nn.Linear(in_features=512, out_features=32, bias=True) \
            ,nn.Softmax())


    def forward(self, x, states):
        # params:
        # x -- the input frame, batch_size = 1
        # states -- the pre-init state for LSTM layer

        x = self.features(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        # x = self.classifier(x)

        x = self.bn(x)

        # Get rid of the first dummy channel dimension
        x = x.unsqueeze(0)

        x, states = self.lstm(x, states)

        # now the output should be of dimensions 1 x batch_size x 32
        x = x.squeeze()

        return self.end(x), states


