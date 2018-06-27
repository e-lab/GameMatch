#
# aac_lstm.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lstm import LSTM
import random


class BaseModelLSTM(nn.Module):
    def __init__(self):
        super(BaseModelLSTM, self).__init__()
        in_channels = 3
        button_num = 32
        
        self.screen_feature_num = 512
        self.feature_num = self.screen_feature_num
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        # self.screen_features1 = LSTM(512 * 2 * 4, self.screen_feature_num)
        self.screen_features1 = LSTM(512 * 4, self.screen_feature_num)

        layer1_size = 128
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size, button_num)


    def forward(self, screen, states):
        # cnn
        screen_features = F.relu(self.conv1(screen))
        screen_features = F.relu(self.conv2(screen_features))
        screen_features = F.relu(self.conv3(screen_features))
        screen_features = F.relu(self.conv4(screen_features))
        screen_features = F.relu(self.conv5(screen_features))
        screen_features = F.relu(self.conv6(screen_features))

        # print the feature dimensions
        # print(screen_features.size())

        screen_features = screen_features.view(screen_features.size(0), -1)

        # print the feature dimensions
        # print(screen_features.size())

        # lstm
        hx, cx = self.screen_features1(screen_features, states)

        # print the output dimensions
        # print(hx.size())

        # action
        action = F.relu(self.action1(hx))
        # action = torch.cat([action, variables], 1)
        action = self.action2(action)
        return action, (hx, cx)

